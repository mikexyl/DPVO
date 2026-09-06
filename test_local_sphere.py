import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from dpvo.local_sphere import LocalSphereBuilder, observed_sphere_mesh, project_points


def make_frame(timestamp):
    x, y = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-0.8, 0.8, 10))
    points = np.stack((x, y, np.full_like(x, 2)), axis=-1).reshape(-1, 3).astype(np.float32)
    return SimpleNamespace(timestamp=timestamp, camera_points=points,
                           colors=np.tile([50, 120, 240], (len(points), 1)).astype(np.uint8),
                           region_ids=np.full(len(points), 20 + timestamp, np.uint16),
                           region_colors=np.tile([200, 80, 30], (len(points), 1)).astype(np.uint8))


class FakeMap:
    def __init__(self):
        self.offset = 0.0

    def pose(self, timestamp, num_frames):
        return np.asarray([-timestamp * 0.1 - self.offset, 0, 0, 0, 0, 0, 1], np.float32)


class LocalSphereTests(unittest.TestCase):
    def project(self, points, timestamps=None, splat_radius=0):
        n = len(points)
        return project_points(np.asarray(points, np.float32),
                              np.arange(n * 3, dtype=np.uint8).reshape(n, 3),
                              np.arange(n, dtype=np.int32) if timestamps is None else np.asarray(timestamps),
                              256, np.arange(1, n + 1, dtype=np.uint16),
                              np.arange(n * 3, dtype=np.uint8).reshape(n, 3), splat_radius)

    def test_zbuffer_preserves_nearest_color_label_and_source(self):
        atlas = self.project([[0, 0, 1], [0, 0, 3]])
        self.assertEqual(atlas['radial_depth'][64, 128], 1)
        self.assertEqual(atlas['region_ids'][64, 128], 1)
        self.assertEqual(atlas['source_timestamp'][64, 128], 0)
        np.testing.assert_array_equal(atlas['rgb'][64, 128], [0, 1, 2, 255])
        self.assertEqual(np.count_nonzero(atlas['rgb'][..., 3]), 1)

    def test_equal_depth_prefers_latest_source(self):
        atlas = self.project([[0, 0, 1], [0, 0, 1]], [10, 20])
        self.assertEqual(atlas['region_ids'][64, 128], 2)
        self.assertEqual(atlas['source_timestamp'][64, 128], 20)

    def test_angular_reduction_matches_naive_point_splat_zbuffer(self):
        rng = np.random.default_rng(7)
        directions = rng.normal(size=(20, 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = np.repeat(directions, 8, axis=0) * rng.uniform(0.5, 4, (160, 1)).astype(np.float32)
        actual = self.project(points, splat_radius=1)
        expected = np.full((128, 256), np.inf, np.float32)
        sources = np.full((128, 256), -1, np.int32)
        for index, point in enumerate(points):
            distance = np.linalg.norm(point)
            u = int(np.floor((np.arctan2(point[0], point[2]) / (2 * np.pi) + 0.5) * 256)) % 256
            v = int(np.clip(np.floor((np.arcsin(np.clip(point[1] / distance, -1, 1)) / np.pi + 0.5) * 128), 0, 127))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    y, x = v + dy, (u + dx) % 256
                    if 0 <= y < 128 and (distance < expected[y, x] or
                                         (distance == expected[y, x] and index > sources[y, x])):
                        expected[y, x], sources[y, x] = distance, index
        np.testing.assert_array_equal(actual['source_timestamp'], sources)
        valid = np.isfinite(expected)
        np.testing.assert_allclose(actual['radial_depth'][valid], expected[valid])

    def test_camera_axes_seam_and_poles(self):
        atlas = self.project([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        self.assertEqual(atlas['region_ids'][64, 192], 1)
        self.assertEqual(atlas['region_ids'][64, 64], 2)
        self.assertEqual(atlas['region_ids'][127, 128], 3)
        self.assertEqual(atlas['region_ids'][0, 128], 4)
        seam = self.project([[0, 0, -1]], splat_radius=1)
        self.assertTrue(np.isfinite(seam['radial_depth'][64, 0]))
        self.assertTrue(np.isfinite(seam['radial_depth'][64, -1]))

    def test_invalid_points_and_unknown_directions(self):
        atlas = self.project([[0, 0, 0], [np.nan, 0, 1], [np.inf, 1, 1]])
        self.assertFalse(atlas['rgb'].any())
        self.assertTrue(np.isnan(atlas['radial_depth']).all())
        self.assertTrue((atlas['source_timestamp'] == -1).all())

    def test_mesh_omits_unobserved_blocks_and_stays_on_display_sphere(self):
        coverage = np.zeros((8, 16), bool)
        coverage[4:6, 8:10] = True
        vertices, triangles, uv = observed_sphere_mesh(coverage, 0.7)
        self.assertEqual(triangles.shape, (2, 3))
        self.assertEqual(vertices.shape, (4, 3))
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 0.7, atol=1e-6)
        forward = np.flatnonzero(np.all(uv == [0.5, 0.5], axis=1))[0]
        np.testing.assert_allclose(vertices[forward], [0, 0, 0.7], atol=1e-6)
        coverage[:] = False
        self.assertEqual(observed_sphere_mesh(coverage, 1)[1].shape, (0, 3))

    def test_sliding_overlap_cadence_bounded_window_and_final_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = LocalSphereBuilder(FakeMap(), directory, window_size=5, every=3, width=256)
            snapshots = []
            for t in range(9):
                snapshots += builder.ingest([make_frame(t)], t + 1)
                self.assertLessEqual(len(builder.window), 5)
            snapshots += builder.finalize(9)
            self.assertEqual([s.anchor_timestamp for s in snapshots], [4, 7, 8])
            self.assertEqual([s.metadata['source_timestamps'] for s in snapshots],
                             [list(range(5)), list(range(3, 8)), list(range(4, 9))])
            self.assertEqual(builder.finalize(9), [])
            self.assertEqual(builder.ingest([make_frame(8)], 9), [])
            self.assertEqual(builder.accepted, 9)
            for sphere in snapshots:
                np.testing.assert_allclose(sphere.center, [sphere.anchor_timestamp * 0.1, 0, 0], atol=1e-6)
                np.testing.assert_allclose(sphere.source_centers[-1], 0, atol=1e-6)
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            self.assertEqual(len(manifest['snapshots']), 3)
            data = np.load(Path(directory) / 'keyframe_000008' / 'projection.npz')
            image = cv2.imread(str(Path(directory) / 'keyframe_000008' / 'rgb.png'), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(image[..., 3] > 0, np.isfinite(data['radial_depth']))
            valid = np.isfinite(data['radial_depth'])
            self.assertTrue(set(data['source_timestamp'][valid]).issubset(set(range(4, 9))))
            np.testing.assert_array_equal(data['region_ids'][valid], data['source_timestamp'][valid] + 20)
            with self.assertRaises(FileExistsError):
                LocalSphereBuilder(FakeMap(), directory)

    def test_recomputes_current_poses_and_allows_small_final_window(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = FakeMap()
            builder = LocalSphereBuilder(dense, directory, width=256)
            self.assertEqual(builder.ingest([make_frame(0), make_frame(1)], 2), [])
            dense.offset = 3.0
            snapshot = builder.finalize(2)[0]
            np.testing.assert_allclose(snapshot.center, [3.1, 0, 0], atol=1e-6)
            self.assertTrue(snapshot.metadata['final'])

    def test_projection_uses_anchor_orientation_not_world_axes(self):
        class RotatedMap:
            def pose(self, timestamp, num_frames):
                if timestamp == 0:
                    return np.asarray([0, 0, 0, 0, 0, 0, 1], np.float32)
                # Camera forward is world +X: world-to-camera rotates -90 around Y.
                return np.asarray([0, 0, 0, 0, -np.sqrt(0.5), 0, np.sqrt(0.5)], np.float32)

        frames = [make_frame(0), make_frame(1)]
        frames[0].camera_points[:] = [2, 0, 0]
        frames[1].camera_points[:] = [0, 0, 2]
        with tempfile.TemporaryDirectory() as directory:
            builder = LocalSphereBuilder(RotatedMap(), directory, window_size=2, width=256)
            sphere = builder.ingest(frames, 2)[0]
            rows, cols = np.where(np.isfinite(sphere.radial_depth))
            self.assertLess(abs(cols.mean() - 128), 2)
            self.assertLess(abs(rows.mean() - 64), 2)
            np.testing.assert_allclose(sphere.radial_depth[rows, cols], 2, atol=1e-6)

    def test_rerun_attaches_mesh_to_anchor_camera(self):
        from dpvo.da3_dense import _camera_to_world
        from dpvo.rerun_viewer import RerunViewer
        with tempfile.TemporaryDirectory() as directory:
            dense = FakeMap()
            builder = LocalSphereBuilder(dense, directory, window_size=2, width=256)
            viewer = object.__new__(RerunViewer)
            viewer.sphere_updates = builder.ingest([make_frame(0), make_frame(1)], 2)
            viewer.sphere_anchor = None
            viewer.dense_map, viewer.num_frames = dense, 2
            viewer._dense_camera_to_world = _camera_to_world
            with patch('dpvo.rerun_viewer.rr.log') as log, patch('dpvo.rerun_viewer.rr.Transform3D') as transform:
                viewer._log_local_sphere()
            np.testing.assert_allclose(transform.call_args.kwargs['translation'], [0.1, 0, 0], atol=1e-6)
            self.assertIn('world/local_sphere/current/mesh', [c.args[0] for c in log.call_args_list])
            self.assertEqual(viewer.sphere_anchor, 1)
            self.assertFalse(viewer.sphere_updates)

    def test_option_validation_and_missing_depth(self):
        from dpvo.rerun_viewer import RerunViewer
        with tempfile.TemporaryDirectory() as directory:
            for options in [dict(window_size=1), dict(window_size=61), dict(every=0), dict(width=258), dict(radius=np.nan), dict(radius=0)]:
                with self.assertRaises(ValueError):
                    LocalSphereBuilder(FakeMap(), directory, **options)
            with self.assertRaisesRegex(ValueError, 'DA3'):
                RerunViewer(SimpleNamespace(), 8, 8, sphere_options=dict(output_dir=directory))

    def test_history_gain_measures_new_directions_not_repeated_observations(self):
        dense = SimpleNamespace(pose=lambda t, n: np.asarray([0, 0, 0, 0, 0, 0, 1], np.float32))
        for different in [False, True]:
            with self.subTest(different=different), tempfile.TemporaryDirectory() as directory:
                frames = [make_frame(0), make_frame(1)]
                frames[0].camera_points[:] = [-1.5 if different else 1.5, 0, 2]
                frames[1].camera_points[:] = [1.5, 0, 2]
                builder = LocalSphereBuilder(dense, directory, window_size=2, width=256)
                sphere = builder.ingest(frames, 2)[0]
                novel = np.isfinite(sphere.radial_depth) & ~np.isfinite(sphere.anchor_radial_depth)
                np.testing.assert_array_equal(sphere.history_gain[..., 3] > 0, novel)
                self.assertAlmostEqual(sphere.metadata['history_only_fraction'], 0.5 if different else 0.0)
                self.assertTrue((sphere.source_timestamp[novel] == 0).all())

    def test_larger_default_window_retains_thirty_keyframes(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = LocalSphereBuilder(FakeMap(), directory, width=256)
            self.assertEqual(builder.ingest([make_frame(t) for t in range(29)], 29), [])
            self.assertEqual(len(builder.ingest([make_frame(29)], 30)), 1)
            builder.ingest([make_frame(30)], 31)
            self.assertEqual(len(builder.window), 30)
            self.assertEqual(builder.window[0].timestamp, 1)


if __name__ == '__main__':
    unittest.main()

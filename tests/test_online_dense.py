"""Synthetic geometry and lifecycle checks for optional online DA3 mapping."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from collections import OrderedDict
from queue import Queue
import numpy as np
from deploy.jetson.dense_mapping import align_scale, stable_camera_pose, RollingCloud, sample_depth, OnlineDenseMapper


class DenseGeometryTest(unittest.TestCase):
    def test_snapshot_uses_mature_input_id_and_stable_landmark_depth(self):
        class Tensor:
            def __init__(self, value): self.value = value
            def __getitem__(self, index): return Tensor(self.value[index])
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        class Pose:
            def __init__(self, value): self.value = value
            def inv(self): return self
            def matrix(self): return self.value
        mapper = object.__new__(OnlineDenseMapper)
        mapper.results, mapper.jobs = Queue(), Queue(maxsize=1)
        mapper.process = NS(is_alive=lambda: True)
        mapper.pending, mapper.last_key = False, None
        mapper.last_submit, mapper.interval = 0., 2.
        mapper.cloud, mapper.cache = RollingCloud(), OrderedDict()
        mapper.remember(104, np.full((24, 32, 3), 17, np.uint8))
        gauge = np.eye(4)
        gauge[:3, :3] *= 2
        graph = NS(tstamps_=np.arange(100, 108),
                   poses_=Tensor(np.tile(np.eye(4), (8, 1, 1))),
                   patches_=Tensor(np.ones((8, 32, 3, 3, 3))),
                   session_from_map_=gauge)
        slam = NS(is_initialized=True, n=8, pg=graph, RES=4)
        # Only the repository's vendored Lie algebra module is available.
        with patch.dict('sys.modules', {'dpvo.lietorch': NS(SE3=Pose)}):
            self.assertIsNone(mapper.update(slam, np.array([20., 20., 16., 12.])))
        key, rgb, intrinsics, uv, target = mapper.jobs.get_nowait()
        self.assertEqual(key, 104)
        self.assertTrue(mapper.pending)
        np.testing.assert_array_equal(uv, np.full((32, 2), 4.))
        np.testing.assert_allclose(target, np.full(32, 2.))
        self.assertEqual(int(rgb[0, 0, 0]), 17)

    def test_robust_scale_rejects_bad_landmarks(self):
        depth = np.full((20, 30), 2., np.float32)
        confidence = np.ones_like(depth)
        uv = np.array([(x, y) for y in range(2, 18, 3) for x in range(2, 28, 3)])
        target = np.full(len(uv), 7.)
        target[:8] = 100
        scale, stats = align_scale(depth, confidence, uv, target)
        self.assertAlmostEqual(scale, 3.5)
        self.assertEqual(stats['inliers'], len(uv) - 8)

    def test_no_alignment_with_invalid_or_insufficient_depth(self):
        depth = np.ones((20, 30))
        uv = np.array([(x, 4) for x in range(20)])
        with self.assertRaises(ValueError):
            align_scale(depth, np.zeros_like(depth), uv, np.ones(20))
        with self.assertRaises(ValueError):
            align_scale(depth, depth, uv[:10], np.ones(10))
        invalid = sample_depth(depth, np.array([[np.nan, 1], [30, 4], [-1, 1]]))
        self.assertTrue(np.isnan(invalid).all())

    def test_stable_gauge_depth_and_pose_cancel_normalization(self):
        camera = np.eye(4)
        camera[:3, 3] = [1, 2, 3]
        gauge = np.eye(4)
        gauge[:3, :3] *= 2
        gauge[:3, 3] = [5, 6, 7]
        scale, pose = stable_camera_pose(gauge, camera)
        camera_point = np.array([.2, .4, 3.])
        expected = gauge[:3, :3] @ (camera_point + camera[:3, 3]) + gauge[:3, 3]
        np.testing.assert_allclose(pose[:3, :3] @ (camera_point * scale) + pose[:3, 3], expected)
        # DPVO map is rescaled by 4; corresponding stable gauge shrinks by 4.
        camera[:3, 3] *= 4
        gauge[:3, :3] /= 4
        next_scale, next_pose = stable_camera_pose(gauge, camera)
        np.testing.assert_allclose(next_pose, pose)
        np.testing.assert_allclose(camera_point * 4 * next_scale, camera_point * scale)

    def test_fusion_updates_poses_culls_and_bounds(self):
        cloud = RollingCloud(max_frames=2, max_points=3, voxel_size=.01)
        xyz = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1]], np.float32)
        rgb = np.zeros((4, 3), np.uint8)
        cloud.add(1, xyz, rgb)
        pose = np.eye(4)
        before, _ = cloud.fused({1: pose})
        self.assertLessEqual(len(before), 3)
        pose[0, 3] = 10
        after, _ = cloud.fused({1: pose})
        np.testing.assert_allclose(after, before + [10, 0, 0])
        cloud.add(2, xyz, rgb)
        cloud.add(3, xyz, rgb)
        self.assertNotIn(1, cloud.frames)
        cloud.fused({3: pose})
        self.assertEqual(list(cloud.frames), [3])
        self.assertEqual(len(cloud.fused({})[0]), 0)


if __name__ == '__main__':
    unittest.main()

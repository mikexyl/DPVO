import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from plyfile import PlyData

from dpvo.da3_dense import DenseMapBuilder
from dpvo.sam_segmenter import masks_to_annotations


class DenseRegionTests(unittest.TestCase):
    def make_builder(self, region_coloring="sam_video_track_id"):
        graph = SimpleNamespace(
            tstamps_=np.array([10, 20]),
            intrinsics_=torch.tensor([[2, 2, 0.875, 0.875]] * 2),
            poses_=torch.tensor([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=torch.float32),
        )
        with patch("dpvo.da3_dense.DA3TensorRT", return_value=SimpleNamespace(height=4, width=4)):
            return DenseMapBuilder("unused.engine", graph, 8, 8, point_stride=1,
                                   region_coloring=region_coloring)

    def annotations(self, region_id=200):
        masks = np.zeros((1, 8, 8), dtype=bool)
        masks[0, :, :4] = True
        return masks_to_annotations(masks, [0.9], (8, 8), 1, [region_id])

    def frame(self, builder, timestamp=10, index=0, depth=None):
        if depth is None:
            depth = np.ones((4, 4), dtype=np.float32)
        return builder._make_keyframe(index, timestamp, depth, np.ones_like(depth),
                                     np.full((4, 4, 3), 123, np.uint8), 2.0, 0.1, 12)

    def test_same_timestamp_integer_ids_colors_and_geometry(self):
        builder = self.make_builder()
        first = self.annotations(200)
        builder.cache_regions(10, first)
        builder.cache_regions(20, self.annotations(9))
        frame = self.frame(builder)
        np.testing.assert_array_equal(frame.region_ids.reshape(4, 4),
                                      [[200, 200, 0, 0]] * 4)
        np.testing.assert_array_equal(frame.region_colors[frame.region_ids == 200],
                                      np.tile(first.colors[0], (8, 1)))
        self.assertTrue(np.all(frame.region_colors[frame.region_ids == 0] == 96))
        self.assertTrue(np.all(frame.colors == 123))
        self.assertTrue(np.all(frame.camera_points[:, 2] == 2))
        self.assertEqual(len(frame.region_ids), len(frame.camera_points))

    def test_invalid_depth_filters_labels_in_lockstep(self):
        builder = self.make_builder()
        builder.cache_regions(10, self.annotations())
        depth = np.ones((4, 4), dtype=np.float32)
        depth[0, 0] = np.nan
        depth[-1, -1] = 0
        frame = self.frame(builder, depth=depth)
        self.assertEqual(len(frame.camera_points), 14)
        self.assertEqual(np.count_nonzero(frame.region_ids), 7)
        self.assertTrue(np.isfinite(frame.camera_points).all())

    def test_noninteger_resize_uses_pixel_centers_without_new_ids(self):
        builder = self.make_builder()
        builder.inference.height, builder.inference.width = 3, 5
        atlas = np.arange(1, 65, dtype=np.uint16).reshape(8, 8)
        annotations = SimpleNamespace(segmentation=atlas, segmentation_context=[])
        builder.cache_regions(10, annotations)
        rows = np.floor((np.arange(3) + 0.5) * 8 / 3).astype(int)
        cols = np.floor((np.arange(5) + 0.5) * 8 / 5).astype(int)
        np.testing.assert_array_equal(builder.regions[10][0], atlas[np.ix_(rows, cols)])

    def test_empty_masks_missing_frame_and_disabled_mode(self):
        builder = self.make_builder()
        with self.assertRaisesRegex(RuntimeError, "same-frame"):
            self.frame(builder)
        builder.cache_regions(10, None)
        frame = self.frame(builder)
        self.assertTrue(np.all(frame.region_ids == 0))
        self.assertTrue(np.all(frame.region_colors == 96))
        builder = self.make_builder(None)
        builder.cache_regions(10, self.annotations())
        self.assertFalse(builder.regions)
        self.assertIsNone(self.frame(builder).region_ids)

    def test_cache_copies_and_prunes_rgb_and_labels_together(self):
        builder = self.make_builder()
        annotations = self.annotations()
        for timestamp in [10, 15, 20]:
            builder.cache_image(timestamp, np.zeros((8, 8, 3), np.uint8))
            builder.cache_regions(timestamp, annotations)
        annotations.segmentation[:] = 0
        self.assertIn(200, builder.regions[10][0])
        builder._prune_cache(15)
        self.assertEqual(set(builder.regions), {15, 20})
        self.assertEqual(set(builder.images), {15, 20})

    def test_ply_preserves_rgb_and_exports_ids_source_frames(self):
        builder = self.make_builder()
        for timestamp, index, region_id in [(10, 0, 200), (20, 1, 9)]:
            builder.cache_regions(timestamp, self.annotations(region_id))
            builder.frames[timestamp] = self.frame(builder, timestamp, index)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.ply"
            metadata = builder.save(path, 2)
            rgb = PlyData.read(path)["vertex"].data
            sam = PlyData.read(path.with_name("map_sam.ply"))["vertex"].data
            self.assertEqual(metadata["points"], 32)
            self.assertEqual(metadata["labeled_points"], 16)
            self.assertEqual(metadata["labeled_fraction"], 0.5)
            self.assertTrue(np.all(rgb["red"] == 123))
            self.assertEqual(set(sam["region_id"]), {0, 9, 200})
            for coord in "xyz":
                np.testing.assert_array_equal(sam[coord], rgb[coord])
            self.assertEqual(set(sam["region_id"][sam["keyframe_timestamp"] == 10]), {0, 200})
            self.assertEqual(set(sam["region_id"][sam["keyframe_timestamp"] == 20]), {0, 9})

    def test_rerun_rgb_and_sam_points_share_poses(self):
        from dpvo.da3_dense import _camera_to_world
        from dpvo.rerun_viewer import RerunViewer

        builder = self.make_builder()
        builder.cache_regions(10, self.annotations())
        frame = self.frame(builder)
        builder.frames[10] = frame
        viewer = object.__new__(RerunViewer)
        viewer.dense_updates = [frame]
        viewer.dense_map = builder
        viewer.num_frames = 2
        viewer._dense_camera_to_world = _camera_to_world
        with patch("dpvo.rerun_viewer.rr.log") as log:
            viewer._log_dense_map()
        paths = [call.args[0] for call in log.call_args_list]
        self.assertIn("world/dense/keyframe_000010/points", paths)
        self.assertIn("world/dense_sam/keyframe_000010/points", paths)
        self.assertIn("world/dense/keyframe_000010", paths)
        self.assertIn("world/dense_sam/keyframe_000010", paths)
        self.assertFalse(viewer.dense_updates)
        self.assertIsNone(frame.aligned_depth)
        self.assertIsNotNone(frame.region_ids)


if __name__ == "__main__":
    unittest.main()

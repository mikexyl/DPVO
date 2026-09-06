import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from dpvo.sam_segmenter import Sam2Segmenter, masks_to_annotations
from dpvo.sam_video import Sam2VideoSegmenter, associate_ids, select_seeds


class SamVideoTests(unittest.TestCase):
    def test_track_ids_colors_survive_area_reordering(self):
        masks = np.zeros((2, 8, 10), dtype=bool)
        masks[0, :3, :3] = True
        masks[1, 4:, 4:] = True
        first = masks_to_annotations(masks, [0.9, 0.95], (8, 10), 1, [22, 11], "presence")
        masks[0, :4, :] = True
        second = masks_to_annotations(masks, [0.9, 0.95], (8, 10), 1, [22, 11], "presence")
        self.assertNotEqual(first.instance_ids.tolist(), second.instance_ids.tolist())
        colors1 = {i: c for i, _, c in first.segmentation_context}
        colors2 = {i: c for i, _, c in second.segmentation_context}
        self.assertEqual(colors1, colors2)
        self.assertEqual(second.segmentation[0, 0], 22)
        self.assertTrue(all("track " in name and "presence" in name for name in second.labels))

    def test_ids_filter_with_small_masks_and_validate(self):
        masks = np.zeros((2, 8, 10), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 2:6, 3:7] = True
        result = masks_to_annotations(masks, [0.9, 0.9], (8, 10), 2, [4, 9])
        np.testing.assert_array_equal(result.instance_ids, [9])
        for ids in ([1, 1], [0, 2], [1, 65536], [1]):
            with self.assertRaises(ValueError):
                masks_to_annotations(masks, [0.9, 0.9], (8, 10), 1, ids)

    def test_sparse_id_preview_uses_track_palette(self):
        model = object.__new__(Sam2Segmenter)
        model.config, model.frames = {}, []
        annotations = masks_to_annotations(np.ones((1, 8, 10)), [0.9], (8, 10), 1, [200])
        image = np.full((8, 10, 3), 100, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            model.save_report(path, image, annotations)
            preview = cv2.imread(str(path.with_suffix(".png")))
            np.testing.assert_array_equal(preview[0, 0],
                (0.5 * image[0, 0] + 0.5 * annotations.colors[0, ::-1]).astype(np.uint8))

    def test_seed_selection_avoids_redundant_nested_parts(self):
        masks = np.zeros((3, 8, 10), dtype=bool)
        masks[0, :4, :4] = True
        masks[1, :2, :2] = True
        masks[2, 4:, 5:] = True
        annotations = masks_to_annotations(masks, [0.9] * 3, (8, 10), 1)
        selected = select_seeds(annotations, 2, 1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(annotations.instance_masks[selected].sum(), 36)
        self.assertEqual(len(select_seeds(None, 8, 1)), 0)

    def test_refresh_association_is_one_to_one_and_keeps_ids(self):
        masks = np.zeros((2, 8, 10), dtype=bool)
        masks[0, :4, :4] = True
        masks[1, 4:, 5:] = True
        previous = masks_to_annotations(masks, [0.9] * 2, (8, 10), 1, [4, 7])
        new = np.stack([masks[1], masks[0], masks[0]])
        ids, next_id = associate_ids(new, previous, 10)
        self.assertEqual(ids[0], 7)
        self.assertEqual(set(ids[1:]), {4, 10})
        self.assertEqual(next_id, 11)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            associate_ids(masks, None, 65536)

    def test_temporal_step_uses_history_and_bounds_memory(self):
        model = object.__new__(Sam2VideoSegmenter)
        model.ids = np.array([1])
        model.model = MagicMock()
        model.model.max_obj_ptrs_in_encoder = 16
        result = dict(maskmem_features="memory", maskmem_pos_enc="position", obj_ptr="pointer",
                      pred_masks_high_res="do not retain this full-resolution raster")
        model.model.track_step.return_value = result
        model._features = MagicMock(return_value=("features", "positions", "sizes"))
        model.outputs = dict(cond_frame_outputs={0: result}, non_cond_frame_outputs={})
        with patch("torch.autocast", return_value=nullcontext()):
            for frame in range(1, 50):
                model._track(frame)
                self.assertLessEqual(len(model.outputs["non_cond_frame_outputs"]), 16)
        call = model.model.track_step.call_args.kwargs
        self.assertFalse(call["is_init_cond_frame"])
        self.assertTrue(call["run_mem_encoder"])
        self.assertIs(call["output_dict"], model.outputs)
        self.assertEqual(call["num_frames"], 50)
        self.assertNotIn("pred_masks_high_res", model.outputs["non_cond_frame_outputs"][49])


if __name__ == "__main__":
    unittest.main()

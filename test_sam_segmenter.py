import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from dpvo.annotations import FrameAnnotations
from dpvo.sam_segmenter import Sam2Segmenter, masks_to_annotations


class SamMaskTests(unittest.TestCase):
    def test_nested_regions_and_background(self):
        masks = np.zeros((2, 8, 10), dtype=bool)
        masks[0, 3:5, 4:6] = True
        masks[1, 1:7, 1:9] = True
        result = masks_to_annotations(masks, [0.92, 0.98], (8, 10), min_mask_area=1)
        self.assertIsInstance(result, FrameAnnotations)
        self.assertEqual(result.segmentation.dtype, np.uint16)
        self.assertEqual(result.segmentation[0, 0], 0)
        self.assertEqual(result.segmentation[2, 2], 1)
        self.assertEqual(result.segmentation[3, 4], 2)
        np.testing.assert_array_equal(result.boxes, [[1, 1, 9, 7], [4, 3, 6, 5]])
        self.assertEqual(result.segmentation_context[0], (0, "unassigned", (0, 0, 0, 0)))
        self.assertEqual(result.class_names, ["region", "region"])
        np.testing.assert_array_equal(result.class_ids, [0, 0])

    def test_resize_preserves_rectangle_coordinates(self):
        masks = np.zeros((1, 4, 4), dtype=bool)
        masks[0, 1:3, 2:4] = True
        result = masks_to_annotations(masks, [0.9], (8, 12), min_mask_area=1)
        np.testing.assert_array_equal(result.boxes, [[6, 2, 12, 6]])
        self.assertEqual(result.segmentation.shape, (8, 12))
        self.assertEqual(np.count_nonzero(result.segmentation), 24)

    def test_filter_after_resizing_and_reject_nonfinite_scores(self):
        masks = np.zeros((3, 4, 4), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, :2, :2] = True
        masks[2] = True
        result = masks_to_annotations(masks, [0.9, 0.95, np.nan], (8, 8), min_mask_area=5)
        self.assertEqual(len(result.scores), 1)
        self.assertEqual(result.segmentation.max(), 1)
        self.assertEqual(result.instance_masks.sum(), 16)

    def test_equal_area_high_quality_wins_overlap(self):
        masks = np.ones((2, 4, 4), dtype=bool)
        result = masks_to_annotations(masks, [0.95, 0.85], (4, 4), min_mask_area=1)
        self.assertTrue(np.all(result.segmentation == 2))
        self.assertAlmostEqual(float(result.scores[-1]), 0.95)

    def test_empty_and_invalid_masks(self):
        self.assertIsNone(masks_to_annotations(np.zeros((0, 4, 4)), [], (4, 4)))
        self.assertIsNone(masks_to_annotations(np.zeros((2, 4, 4)), [0.9, 0.9], (4, 4), 0))
        with self.assertRaises(ValueError):
            masks_to_annotations(np.zeros((2, 4, 4)), [0.9], (4, 4))
        with self.assertRaises(ValueError):
            masks_to_annotations(np.zeros((2, 4, 4)), [0.9, 0.9], (0, 4))

    def test_report_and_bgr_preview(self):
        model = object.__new__(Sam2Segmenter)
        model.config = {"frame_local_ids": True}
        model.frames = [{"frame": 0, "masks": 1, "coverage": 1.0, "seconds": 0.5}]
        annotations = masks_to_annotations(np.ones((1, 8, 10)), [0.9], (8, 10), 1)
        image = np.full((8, 10, 3), 100, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            summary = model.save_report(path, image, annotations)
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(json.loads(path.read_text())["summary"]["mean_masks"], 1)
            import cv2
            preview = cv2.imread(str(path.with_suffix(".png")))
            expected = (0.5 * image[0, 0] + 0.5 * annotations.colors[0, ::-1]).astype(np.uint8)
            np.testing.assert_array_equal(preview[0, 0], expected)


class SamViewerTests(unittest.TestCase):
    def test_sam_only_does_not_initialize_yolo_or_depth(self):
        from dpvo.rerun_viewer import RerunViewer

        yolo, depth = MagicMock(), MagicMock()
        with patch.dict("sys.modules", {"dpvo.yolo_detector": yolo, "dpvo.da3_dense": depth}), \
                patch("dpvo.sam_segmenter.Sam2Segmenter") as sam, \
                patch("dpvo.rerun_viewer.rr") as rr:
            viewer = RerunViewer(SimpleNamespace(), 8, 10, sam_model="sam2.1_hiera_tiny.pt")
            self.assertIs(viewer.detector, sam.return_value)
            self.assertIsNone(viewer.dense_map)
            self.assertIsNone(viewer.scene_graph)
            yolo.YoloTensorRTDetector.assert_not_called()
            depth.DenseMapBuilder.assert_not_called()
            sam.assert_called_once()
            rr.send_blueprint.assert_called_once()
            viewer.join()

    def test_mask_logging_clears_an_empty_following_frame(self):
        from dpvo.rerun_viewer import RerunViewer

        with patch("dpvo.sam_segmenter.Sam2Segmenter") as sam, \
                patch("dpvo.rerun_viewer.rr") as rr:
            sam.return_value.last_stats = None
            viewer = RerunViewer(SimpleNamespace(), 8, 10, sam_model="sam2.1_hiera_tiny.pt")
            viewer.image = np.zeros((8, 10, 3), dtype=np.uint8)
            viewer.intrinsics = np.array([10, 10, 5, 4])
            viewer.detections = masks_to_annotations(np.ones((1, 8, 10)), [0.9], (8, 10), 1)
            viewer._log_state(0)
            rr.SegmentationImage.assert_called_once()
            rr.AnnotationContext.assert_called_once()
            viewer.detections = None
            rr.reset_mock()
            viewer._log_state(1)
            rr.SegmentationImage.assert_not_called()
            rr.Clear.assert_any_call(recursive=True)
            rr.log.assert_any_call("world/camera/image/segmentation", rr.Clear.return_value)
            viewer.join()

    def test_semantic_scene_graph_and_yolo_are_rejected_with_sam(self):
        from dpvo.rerun_viewer import RerunViewer

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            RerunViewer(SimpleNamespace(), 8, 10, sam_model="sam2.1_hiera_tiny.pt", yolo_model="x.engine")
        with self.assertRaisesRegex(ValueError, "semantic scene graph"):
            RerunViewer(SimpleNamespace(), 8, 10, sam_model="sam2.1_hiera_tiny.pt", scene_graph=True)


class SamTensorRTTests(unittest.TestCase):
    def test_manifest_selects_tensorrt_without_loading_pytorch_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sam2.json"
            path.write_text("{}")
            with patch("dpvo.sam_tensorrt.Sam2TensorRT") as trt, \
                    patch("ultralytics.models.sam.build.build_sam2_t") as build:
                model = Sam2Segmenter(path, points_per_side=16)
                self.assertIs(model.trt, trt.return_value)
                self.assertIs(model.config, trt.return_value.config)
                self.assertFalse(hasattr(model, "predictor"))
                build.assert_not_called()
                trt.assert_called_once_with(path, 16, 100, 0.8, 0.92)

    def test_invalid_manifest_is_rejected_before_gpu_initialization(self):
        from dpvo.sam_tensorrt import Sam2TensorRT
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"format":"not_sam"}')
            with self.assertRaisesRegex(ValueError, "unsupported"):
                Sam2TensorRT(path, 16, 100, 0.8, 0.92)

    def test_gpu_postprocess_algorithm_on_cpu_empty_and_duplicate_proposals(self):
        import torch
        from dpvo.sam_tensorrt import Sam2TensorRT

        runner = object.__new__(Sam2TensorRT)
        runner.iou_thresh, runner.stability_thresh, runner.min_mask_area = 0.8, 0.5, 1
        logits = torch.full((2, 8, 8), -3.0)
        logits[:, 2:6, 2:6] = 3.0
        self.assertIsNone(runner.postprocess(logits, torch.tensor([0.2, 0.3]), (16, 24)))
        result = runner.postprocess(logits, torch.tensor([0.9, 0.95]), (16, 24))
        self.assertEqual(len(result.scores), 1)
        self.assertAlmostEqual(float(result.scores[0]), 0.95)
        self.assertEqual(result.segmentation.shape, (16, 24))
        self.assertEqual(result.segmentation[0, 0], 0)
        self.assertEqual(result.segmentation[8, 12], 1)
        self.assertIsNone(runner.postprocess(torch.full_like(logits, -3), torch.tensor([0.9, 0.95]), (16, 24)))


if __name__ == "__main__":
    unittest.main()

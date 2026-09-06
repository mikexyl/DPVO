import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F

from .annotations import FrameAnnotations

# Keep existing scene-graph callers and saved experiment scripts compatible.
YoloAnnotations = FrameAnnotations


def _load_tensorrt_bindings():
    """Expose NVIDIA's split CUDA bindings under the module name Ultralytics uses."""
    try:
        import tensorrt  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "tensorrt":
            raise

        import tensorrt_bindings

        sys.modules["tensorrt"] = tensorrt_bindings


class YoloTensorRTDetector:
    """Run an Ultralytics detection engine and return Rerun-ready annotations."""

    _COLORS = np.asarray(
        [
            [255, 99, 71],
            [0, 191, 255],
            [50, 205, 50],
            [255, 215, 0],
            [186, 85, 211],
            [255, 140, 0],
            [64, 224, 208],
            [255, 105, 180],
        ],
        dtype=np.uint8,
    )

    def __init__(self, model_path, confidence=0.25, image_size=640, task=None):
        model_path = Path(model_path)
        if model_path.suffix != ".engine":
            raise ValueError(f"YOLO model must be a TensorRT .engine file: {model_path}")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")

        task = task or ("segment" if "-seg" in model_path.stem else "detect")
        if task not in {"detect", "segment"}:
            raise ValueError(f"Unsupported YOLO task: {task}")

        _load_tensorrt_bindings()
        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task=task)
        self.confidence = confidence
        self.image_size = image_size
        self.task = task

    def __call__(self, bgr_image):
        predict_args = dict(
            source=np.ascontiguousarray(bgr_image),
            conf=self.confidence,
            imgsz=self.image_size,
            device=0,
            verbose=False,
        )
        if self.task == "segment":
            predict_args["retina_masks"] = True

        result = self.model.predict(**predict_args)[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        scores = result.boxes.conf.detach().cpu().numpy()
        class_names = [result.names[int(class_id)] for class_id in class_ids]
        labels = [
            f"{class_name} {score:.2f}"
            for class_name, score in zip(class_names, scores)
        ]
        colors = self._COLORS[class_ids % len(self._COLORS)]

        segmentation = None
        segmentation_context = None
        instance_masks = None
        if result.masks is not None:
            masks = result.masks.data
            if tuple(masks.shape[-2:]) != tuple(bgr_image.shape[:2]):
                masks = F.interpolate(
                    masks[None],
                    size=bgr_image.shape[:2],
                    mode="nearest",
                )[0]
            masks = masks.detach().cpu().numpy() > 0.5
            instance_masks = masks

            segmentation = np.zeros(bgr_image.shape[:2], dtype=np.uint16)
            # Results are confidence-sorted. Paint lower-confidence masks first so
            # higher-confidence instances own overlapping pixels.
            for index in reversed(range(len(masks))):
                segmentation[masks[index]] = class_ids[index] + 1

            segmentation_context = []
            for class_id in np.unique(class_ids):
                color = self._COLORS[class_id % len(self._COLORS)]
                segmentation_context.append(
                    (
                        int(class_id) + 1,
                        result.names[int(class_id)],
                        tuple(int(channel) for channel in color),
                    )
                )

        return YoloAnnotations(
            boxes=boxes,
            class_ids=class_ids,
            class_names=class_names,
            scores=scores,
            labels=labels,
            colors=colors,
            image_shape=tuple(bgr_image.shape[:2]),
            instance_masks=instance_masks,
            segmentation=segmentation,
            segmentation_context=segmentation_context,
        )

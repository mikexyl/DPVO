"""Class-agnostic SAM 2.1 Hiera Tiny masks on every processed DPVO image.

Supports exported TensorRT bundles and the reference Ultralytics SAM2 runtime
with Meta's official weights. Region IDs are local to each frame, not tracked.
"""

import colorsys
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .annotations import FrameAnnotations


def masks_to_annotations(masks, scores, image_shape, min_mask_area=100,
                         instance_ids=None, score_kind="mask IoU"):
    """Resize proposals and compose an instance atlas, preserving nested regions."""
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float32)
    height, width = image_shape
    if masks.ndim != 3 or scores.shape != (len(masks),):
        raise ValueError("expected masks [N,H,W] and scores [N]")
    ids = None if instance_ids is None else np.asarray(instance_ids, dtype=np.int64)
    if ids is not None and (ids.shape != (len(masks),) or len(np.unique(ids)) != len(ids)
                            or np.any(ids < 1) or np.any(ids > np.iinfo(np.uint16).max)):
        raise ValueError("instance IDs must be unique positive uint16 values, one per mask")
    if height <= 0 or width <= 0 or min_mask_area < 0:
        raise ValueError("image size must be positive and minimum mask area nonnegative")
    if not len(masks):
        return None
    if masks.shape[1:] != (height, width):
        masks = np.stack([
            cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            for mask in masks
        ])
    masks = masks.astype(bool)
    areas = masks.sum(axis=(1, 2))
    keep = (areas >= max(1, min_mask_area)) & np.isfinite(scores)
    masks, scores, areas = masks[keep], scores[keep], areas[keep]
    if ids is not None:
        ids = ids[keep]
    if not len(masks):
        return None
    if len(masks) > np.iinfo(np.uint16).max:
        raise ValueError("too many regions for a uint16 segmentation atlas")

    # Large surfaces first, then smaller objects/parts. For equal areas the
    # higher-quality proposal owns the overlap. Do not fill unobserved pixels.
    order = np.lexsort((scores, -areas.astype(np.int64)))
    masks, scores = masks[order], scores[order]
    if ids is not None:
        ids = ids[order]
    atlas = np.zeros((height, width), dtype=np.uint16)
    boxes, colors, labels = [], [], []
    context = [(0, "unassigned", (0, 0, 0, 0))]
    for index, (mask, score) in enumerate(zip(masks, scores), start=1):
        region_id = index if ids is None else int(ids[index - 1])
        atlas[mask] = region_id
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        boxes.append([x, y, x + w, y + h])
        color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb((region_id * 0.618034) % 1, 0.7, 1.0))
        colors.append(color)
        name = f"region {region_id}" if ids is None else f"track {region_id}"
        labels.append(f"{name}, {score_kind} {score:.2f}")
        context.append((region_id, name + (" (frame-local)" if ids is None else ""), color))

    return FrameAnnotations(
        boxes=np.asarray(boxes, dtype=np.float32),
        # All regions have an unknown semantic class, not the synthetic SAM IDs.
        class_ids=np.zeros(len(masks), dtype=np.int64),
        class_names=["region"] * len(masks),
        scores=scores,
        labels=labels,
        colors=np.asarray(colors, dtype=np.uint8),
        image_shape=(height, width),
        instance_masks=masks,
        segmentation=atlas,
        segmentation_context=context,
        instance_ids=ids,
    )


class Sam2Segmenter:
    """Automatic point-grid proposals; TensorRT or reference FP32, no tracking."""

    def __init__(self, model_path, points_per_side=32, points_per_batch=16,
                 min_mask_area=100, pred_iou_thresh=0.8, stability_score_thresh=0.92):
        model_path = Path(model_path)
        if model_path.suffix != ".json" and model_path.name not in {"sam2.1_hiera_tiny.pt", "sam2.1_t.pt"}:
            raise ValueError("SAM model must be a TensorRT bundle .json or SAM 2.1 Tiny checkpoint")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if points_per_side < 1 or points_per_batch < 1 or min_mask_area < 0:
            raise ValueError("SAM grid/batch sizes must be positive and mask area nonnegative")
        if not 0 <= pred_iou_thresh <= 1 or not 0 <= stability_score_thresh <= 1:
            raise ValueError("SAM quality thresholds must be between zero and one")

        self.frames = []
        self.last_stats = None
        self.trt = None
        if model_path.suffix == ".json":
            from .sam_tensorrt import Sam2TensorRT

            self.trt = Sam2TensorRT(model_path, points_per_side, min_mask_area,
                                   pred_iou_thresh, stability_score_thresh)
            self.config = self.trt.config
            return

        from ultralytics.models.sam.build import build_sam2_t
        from ultralytics.models.sam.predict import SAM2Predictor

        self.predictor = SAM2Predictor(overrides=dict(
            device=0, imgsz=1024, conf=pred_iou_thresh, iou=0.7,
            verbose=False, save=False,
        ))
        # Strict state-dict loading also verifies SAM 2.1 vs SAM 2 architecture.
        self.predictor.setup_model(model=build_sam2_t(str(model_path)), verbose=False)
        self.config = dict(
            model=str(model_path), backend="ultralytics_sam2_pytorch_fp32",
            points_per_side=points_per_side, points_per_batch=points_per_batch,
            min_mask_area=min_mask_area, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            image_size=1024, frame_local_ids=True,
        )

    def __call__(self, bgr_image):
        import torch

        start = perf_counter()
        if self.trt is not None:
            annotations = self.trt(bgr_image)
            torch.cuda.synchronize()
            self._record_stats(annotations, start)
            return annotations
        # SAM2's official image transform is a square resize. Supplying a square
        # source also avoids SAM1 letterboxing in the shared Ultralytics helper.
        square = cv2.resize(bgr_image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        with torch.inference_mode():
            result = self.predictor(
                source=np.ascontiguousarray(square),
                points_stride=self.config["points_per_side"],
                points_batch_size=self.config["points_per_batch"],
                conf_thres=self.config["pred_iou_thresh"],
                stability_score_thresh=self.config["stability_score_thresh"],
                crop_n_layers=0,
            )[0]
        annotations = None
        if result.masks is not None:
            annotations = masks_to_annotations(
                result.masks.data.detach().cpu().numpy(),
                result.boxes.conf.detach().cpu().numpy(),
                bgr_image.shape[:2], self.config["min_mask_area"],
            )
        # Includes preprocessing, CPU transfer and mask composition; synchronize
        # even empty results so the wall-time metric includes GPU work.
        torch.cuda.synchronize(self.predictor.device)
        self._record_stats(annotations, start)
        return annotations

    def _record_stats(self, annotations, start):
        self.last_stats = dict(
            frame=len(self.frames),
            masks=0 if annotations is None else len(annotations.scores),
            coverage=0.0 if annotations is None else float(np.mean(annotations.segmentation > 0)),
            seconds=perf_counter() - start,
        )
        self.frames.append(self.last_stats)
        if len(self.frames) == 1 or len(self.frames) % 30 == 0:
            stats = self.last_stats
            print(f"SAM2 frame {stats['frame']}: {stats['masks']} regions, "
                  f"{stats['coverage']:.1%} coverage, {stats['seconds']:.2f}s", flush=True)

    def save_report(self, path, image=None, annotations=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = dict(
            frames=len(self.frames),
            mean_masks=float(np.mean([f["masks"] for f in self.frames])) if self.frames else 0.0,
            mean_coverage=float(np.mean([f["coverage"] for f in self.frames])) if self.frames else 0.0,
            mean_seconds=float(np.mean([f["seconds"] for f in self.frames])) if self.frames else 0.0,
        )
        path.write_text(json.dumps(dict(config=self.config, summary=summary, frames=self.frames), indent=2))
        if image is not None and annotations is not None:
            atlas = annotations.segmentation
            palette = np.zeros((int(atlas.max()) + 1, 3), dtype=np.uint8)
            for region_id, _, color in annotations.segmentation_context:
                if region_id < len(palette):
                    palette[region_id] = color[:3]
            overlay = image.copy()
            covered = atlas > 0
            overlay[covered] = (0.5 * image[covered] + 0.5 * palette[atlas[covered]][:, ::-1]).astype(np.uint8)
            if not cv2.imwrite(str(path.with_suffix(".png")), overlay):
                raise OSError(f"could not write SAM preview next to {path}")
        return summary

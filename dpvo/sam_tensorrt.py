"""GPU-resident, CUDA-graph SAM 2.1 image encoder and batched mask decoder."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .sam_segmenter import masks_to_annotations
from .yolo_detector import _load_tensorrt_bindings


class StaticTensorRT:
    """Fixed-shape engine with persistent buffers, optionally shared with a producer."""

    def __init__(self, path, shared=None, logger=None):
        _load_tensorrt_bindings()
        import tensorrt as trt

        self.logger = logger if logger is not None else trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not load TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"could not create TensorRT context: {path}")
        self.tensors = {}
        self.inputs = set()
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"SAM engine requires static shapes: {name}={shape}")
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            tensor = (shared or {}).get(name)
            if tensor is None:
                tensor = torch.empty(shape, dtype=dtype, device="cuda")
            if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"incompatible shared SAM tensor: {name}")
            self.tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.add(name)

    def enqueue(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("SAM TensorRT execution failed")


class Sam2TensorRT:
    """Same 1024-pixel, multimask point-grid inference as the PyTorch baseline.

Both neural networks run in TensorRT. Torch CUDA kernels handle postprocessing;
no PyTorch model or checkpoint is loaded at inference time. Full-resolution mask
stability and NMS preserve the baseline's filtering semantics.
"""

    def __init__(self, manifest_path, points_per_side, min_mask_area,
                 pred_iou_thresh, stability_score_thresh, cuda_graph=True):
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        if manifest.get("format") != "dpvo_sam2_tensorrt_v1":
            raise ValueError("unsupported SAM TensorRT manifest")
        if manifest["image_size"] != 1024 or manifest["batch_size"] < 1:
            raise ValueError("invalid SAM TensorRT image/batch dimensions")
        self.encoder = StaticTensorRT(path.parent / manifest["encoder"])
        self.decoder = StaticTensorRT(path.parent / manifest["decoder"], shared=self.encoder.tensors,
                                      logger=self.encoder.logger)
        if set(self.encoder.tensors) != {"images", "image_embed", "high_res_0", "high_res_1"}:
            raise ValueError("unexpected SAM encoder bindings")
        if set(self.decoder.tensors) != {"points", "image_embed", "high_res_0", "high_res_1", "masks", "scores"}:
            raise ValueError("unexpected SAM decoder bindings")
        self.batch_size = manifest["batch_size"]
        if tuple(self.decoder.tensors["points"].shape) != (self.batch_size, 1, 2):
            raise ValueError("SAM manifest batch does not match decoder")
        self.num_prompts = points_per_side ** 2
        self.steps = (self.num_prompts + self.batch_size - 1) // self.batch_size
        axis = (np.arange(points_per_side, dtype=np.float32) + 0.5) * (1024 / points_per_side)
        xx, yy = np.meshgrid(axis, axis)
        grid = np.stack([xx, yy], axis=-1).reshape(-1, 1, 2)
        grid = np.pad(grid, ((0, self.steps * self.batch_size - len(grid)), (0, 0), (0, 0)), mode="edge")
        self.grid = torch.from_numpy(grid).cuda()
        self.logits = torch.empty((self.steps * self.batch_size, 3, 256, 256), device="cuda",
                                  dtype=self.decoder.tensors["masks"].dtype)
        self.scores = torch.empty((self.steps * self.batch_size, 3), device="cuda")
        self.config = dict(
            backend=f"tensorrt_{manifest['precision']}", model=str(path),
            encoder=manifest["encoder"], decoder=manifest["decoder"],
            image_size=1024, points_per_side=points_per_side,
            points_per_batch=self.batch_size, min_mask_area=min_mask_area,
            pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh,
            frame_local_ids=True, cuda_graph=cuda_graph,
            fp16_io=manifest.get("fp16_io", False),
            postprocess="full_resolution_stability_gpu_nms", checkpoint_sha256=manifest["checkpoint_sha256"],
        )
        self.min_mask_area = min_mask_area
        self.iou_thresh = pred_iou_thresh
        self.stability_thresh = stability_score_thresh
        self.mean = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1)
        self.graph = None
        # Capture only static neural inference. Postprocessing remains dynamic;
        # shared encoder outputs and decoder inputs never round-trip through CPU.
        self.encoder.tensors["images"].zero_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            self._enqueue_network()
            self._enqueue_network()
        stream.synchronize()
        if cuda_graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
                self._enqueue_network()
            stream.synchronize()

    def _enqueue_network(self):
        self.encoder.enqueue()
        for step in range(self.steps):
            start = step * self.batch_size
            end = start + self.batch_size
            self.decoder.tensors["points"].copy_(self.grid[start:end])
            self.decoder.enqueue()
            self.logits[start:end].copy_(self.decoder.tensors["masks"])
            self.scores[start:end].copy_(self.decoder.tensors["scores"])

    def infer_raw(self, bgr_image):
        """Return GPU logits/quality scores for validation; overwritten next frame."""
        square = cv2.resize(bgr_image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        rgb = torch.from_numpy(np.ascontiguousarray(square[..., ::-1])).cuda()
        normalized = (rgb.permute(2, 0, 1).unsqueeze(0).float() - self.mean) / self.std
        self.encoder.tensors["images"].copy_(normalized)
        if self.graph is None:
            self._enqueue_network()
        else:
            self.graph.replay()
        return self.logits[:self.num_prompts].flatten(0, 1), self.scores[:self.num_prompts].flatten()

    def postprocess(self, logits, scores, image_shape):
        from torchvision.ops import nms
        from ultralytics.models.sam.amg import batched_mask_to_box, calculate_stability_score

        indices = torch.where((scores > self.iou_thresh) & torch.isfinite(scores))[0]
        if not len(indices):
            return None
        selected = logits[indices]
        stable, boxes = [], []
        # Bound peak GPU memory independently of the number of prompt proposals.
        for chunk in selected.split(32):
            full = F.interpolate(chunk[None], (1024, 1024), mode="bilinear", align_corners=False)[0]
            stable.append(calculate_stability_score(full, 0.0, 0.95) > self.stability_thresh)
            boxes.append(batched_mask_to_box(full > 0).float())
        valid = torch.cat(stable)
        boxes = torch.cat(boxes)[valid]
        indices = indices[valid]
        keep = nms(boxes, scores[indices], 0.7)
        indices = indices[keep]
        if not len(indices):
            return None
        # Match the reference: threshold at 1024, then nearest-neighbor resize
        # the binary masks to the rectangular DPVO image.
        masks = []
        for chunk in logits[indices].split(32):
            full = F.interpolate(chunk[None], (1024, 1024), mode="bilinear", align_corners=False)[0]
            resized = F.interpolate((full > 0).float()[None], image_shape, mode="nearest")[0] > 0
            masks.append(resized)
        masks = torch.cat(masks).cpu().numpy()
        return masks_to_annotations(masks, scores[indices].cpu().numpy(), image_shape, self.min_mask_area)

    @torch.inference_mode()
    def __call__(self, bgr_image):
        logits, scores = self.infer_raw(bgr_image)
        return self.postprocess(logits, scores, bgr_image.shape[:2])

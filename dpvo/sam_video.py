"""Streaming SAM 2.1 video memory on consecutive stride-selected DPVO frames.

Hybrid backend: TensorRT encoder/automatic seeding, original SAM2 track_step
with BF16 PyTorch temporal attention, mask heads and memory encoder. This is
not a fully TensorRT video tracker.
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .sam_segmenter import Sam2Segmenter, masks_to_annotations


def select_seeds(annotations, max_tracks, min_area):
    """Favor large, high-quality regions without spending tracks on nested parts."""
    if annotations is None or annotations.instance_masks is None:
        return np.empty(0, dtype=np.int64)
    masks = annotations.instance_masks
    areas = masks.sum((1, 2))
    order = np.argsort(-(areas * annotations.scores), kind="stable")
    covered = np.zeros(annotations.image_shape, dtype=bool)
    selected = []
    for index in order:
        novel = np.count_nonzero(masks[index] & ~covered)
        if novel < min_area or novel < 0.15 * areas[index]:
            continue
        selected.append(int(index))
        covered |= masks[index]
        if len(selected) >= max_tracks:
            break
    return np.asarray(selected, dtype=np.int64)


def associate_ids(masks, previous, next_id, threshold=0.3):
    """One-to-one current-frame IoU association only at explicit memory refreshes."""
    ids = np.zeros(len(masks), dtype=np.int64)
    if previous is not None and previous.instance_ids is not None:
        # Downsample only association, not the masks used by the video network.
        small = np.stack([cv2.resize(m.astype(np.uint8), (160, 120), interpolation=cv2.INTER_NEAREST)
                          for m in masks]).astype(bool) if len(masks) else np.empty((0, 120, 160), bool)
        old = np.stack([cv2.resize(m.astype(np.uint8), (160, 120), interpolation=cv2.INTER_NEAREST)
                        for m in previous.instance_masks]).astype(bool)
        candidates = []
        for i, mask in enumerate(small):
            overlap = (old & mask).sum((1, 2)) / np.maximum(1, (old | mask).sum((1, 2)))
            candidates += [(float(score), i, j) for j, score in enumerate(overlap) if score >= threshold]
        used = set()
        for _, i, j in sorted(candidates, reverse=True):
            if not ids[i] and j not in used:
                ids[i] = previous.instance_ids[j]
                used.add(j)
    for i in range(len(ids)):
        if not ids[i]:
            if next_id > np.iinfo(np.uint16).max:
                raise RuntimeError("SAM video exhausted the uint16 track ID space")
            ids[i], next_id = next_id, next_id + 1
    return ids, next_id


class Sam2VideoSegmenter(Sam2Segmenter):
    """One streaming state per video; prompt once, then use learned temporal memory."""

    def __init__(self, model_path, points_per_side=16, min_mask_area=100,
                 pred_iou_thresh=0.8, stability_score_thresh=0.92,
                 max_tracks=8, refresh_interval=10, memory_frames=3):
        if max_tracks < 1 or refresh_interval < 0 or not 2 <= memory_frames <= 7:
            raise ValueError("positive track limit, nonnegative refresh interval and 2-7 memories required")
        manifest_path = Path(model_path)
        manifest = json.loads(manifest_path.read_text())
        checkpoint = Path(manifest["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
            raise ValueError("video checkpoint does not match the TensorRT encoder weights")
        from ultralytics.models.sam.build import build_sam2_t

        self.proposer = Sam2Segmenter(manifest_path, points_per_side=points_per_side,
                                      min_mask_area=min_mask_area, pred_iou_thresh=pred_iou_thresh,
                                      stability_score_thresh=stability_score_thresh)
        self.engine = self.proposer.trt
        self.model = build_sam2_t(str(checkpoint)).cuda().eval()
        if memory_frames != self.model.num_maskmem:
            # Preserve the learned recency slots and the conditioning-frame slot.
            original = self.model.maskmem_tpos_enc
            self.model.maskmem_tpos_enc = torch.nn.Parameter(
                torch.cat([original[:memory_frames - 1], original[-1:]]).detach(), requires_grad=False)
        self.model.num_maskmem = memory_frames
        self.model.set_binarize(True)
        self.config = dict(self.proposer.config, mode="video", frame_local_ids=False,
                           backend="tensorrt_encoder_seed_pytorch_bf16_video_memory",
                           temporal_backend="pytorch_bf16", max_tracks=max_tracks,
                           refresh_interval=refresh_interval, memory_frames=memory_frames,
                           refresh_association_iou=0.3, cuda_graph=False,
                           encoder_cuda_graph=True, seed_cuda_graph=True)
        self.frames = []
        self.last_stats = None
        self.ids = np.empty(0, dtype=np.int64)
        self.next_id = 1
        self.outputs = dict(cond_frame_outputs={}, non_cond_frame_outputs={})
        self.last_annotations = None
        self.min_area = min_mask_area
        self.pos = self.model.image_encoder.neck.position_encoding(
            torch.zeros((1, 256, 64, 64), device="cuda")
        ).flatten(2).permute(2, 0, 1)
        # Only temporal/head modules are needed from the checkpoint at runtime.
        # The original image encoder is replaced by TensorRT.
        self.model.image_encoder = None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.engine.encoder.enqueue()
        stream.synchronize()
        self.encoder_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.encoder_graph, stream=stream):
            self.engine.encoder.enqueue()
        stream.synchronize()

    def _features(self, count):
        tensors = self.engine.encoder.tensors
        # Image-only TensorRT export adds no_mem_embed; video memory expects the
        # raw backbone tensor and adds/fuses its own memory exactly once.
        raw = tensors["image_embed"].float() - self.model.no_mem_embed.reshape(1, 256, 1, 1)
        features = [tensors["high_res_0"].float(), tensors["high_res_1"].float(), raw]
        features = [feat.flatten(2).permute(2, 0, 1).expand(-1, count, -1) for feat in features]
        return features, [self.pos.expand(-1, count, -1)], [(256, 256), (128, 128), (64, 64)]

    def _track(self, frame_index, seed_masks=None):
        features, positions, sizes = self._features(len(self.ids))
        initial = seed_masks is not None
        masks = None
        if initial:
            masks = torch.from_numpy(seed_masks.astype(np.float32)).cuda()[:, None]
            masks = F.interpolate(masks, (1024, 1024), mode="nearest")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.track_step(
                frame_idx=frame_index, is_init_cond_frame=initial,
                current_vision_feats=features, current_vision_pos_embeds=positions,
                feat_sizes=sizes, point_inputs=None, mask_inputs=masks,
                output_dict=self.outputs, num_frames=frame_index + 1,
                run_mem_encoder=True,
            )
        # Retain only the fields used for temporal attention/object pointers.
        stored = {key: output[key] for key in ("maskmem_features", "maskmem_pos_enc", "obj_ptr")}
        storage = "cond_frame_outputs" if initial else "non_cond_frame_outputs"
        self.outputs[storage][frame_index] = stored
        # Original SAM uses up to 16 recent object pointers and 7 mask memories.
        for old in list(self.outputs["non_cond_frame_outputs"]):
            if old < frame_index - self.model.max_obj_ptrs_in_encoder + 1:
                del self.outputs["non_cond_frame_outputs"][old]
        return output

    def _annotations(self, output, image_shape):
        logits = output["pred_masks_high_res"].float()
        masks = F.interpolate(logits, image_shape, mode="bilinear", align_corners=False)[:, 0] > 0
        presence = output["object_score_logits"].float().sigmoid().flatten()
        masks &= (presence > 0.5)[:, None, None]
        return masks_to_annotations(masks.cpu().numpy(), presence.cpu().numpy(), image_shape,
                                    self.min_area, instance_ids=self.ids, score_kind="presence")

    @torch.inference_mode()
    def __call__(self, image):
        start = perf_counter()
        index = len(self.frames)
        refresh = not len(self.ids) or (self.config["refresh_interval"] > 0
                                        and index % self.config["refresh_interval"] == 0)
        previous = None
        # At refreshes propagate first, so ID association compares masks in the
        # same image, not unwarped masks from the preceding camera position.
        if len(self.ids):
            square = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            rgb = torch.from_numpy(np.ascontiguousarray(square[..., ::-1])).cuda()
            normalized = (rgb.permute(2, 0, 1)[None].float() - self.engine.mean) / self.engine.std
            self.engine.encoder.tensors["images"].copy_(normalized)
            self.encoder_graph.replay()
            previous = self._annotations(self._track(index), image.shape[:2])
        annotations = previous
        if refresh:
            proposals = self.proposer(image)
            selected = select_seeds(proposals, self.config["max_tracks"], self.min_area)
            if len(selected):
                masks = proposals.instance_masks[selected]
                self.ids, self.next_id = associate_ids(masks, previous, self.next_id)
                self.outputs = dict(cond_frame_outputs={}, non_cond_frame_outputs={})
                annotations = self._annotations(self._track(index, masks), image.shape[:2])
            # If discovery finds nothing, continue existing memory/tracks.
        self.last_annotations = annotations
        torch.cuda.synchronize()
        self._record_stats(annotations, start)
        self.last_stats.update(
            refresh=refresh, active_tracks=len(self.ids),
            visible_ids=[] if annotations is None else annotations.instance_ids.tolist(),
            memory_records=sum(len(frames) for frames in self.outputs.values()),
        )
        return annotations

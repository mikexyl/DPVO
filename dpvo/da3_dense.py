"""Two-view Depth Anything 3 TensorRT inference and DPVO scale alignment."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from .yolo_detector import _load_tensorrt_bindings


def _quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _camera_to_world(pose):
    rotation_camera_from_world = _quaternion_to_matrix(pose[3:])
    rotation_world_from_camera = rotation_camera_from_world.T
    translation_world_from_camera = -rotation_world_from_camera @ pose[:3]
    return rotation_world_from_camera, translation_world_from_camera


class DA3TensorRT:
    """Fixed-shape TensorRT runner for two-view DA3 depth and confidence."""

    _MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, engine_path):
        engine_path = Path(engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)

        _load_tensorrt_bindings()
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize DA3 engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        expected_names = {"images", "depth", "confidence"}
        names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }
        if names != expected_names:
            raise ValueError(f"unexpected DA3 engine tensors: {sorted(names)}")

        self.input_shape = tuple(self.engine.get_tensor_shape("images"))
        self.output_shape = tuple(self.engine.get_tensor_shape("depth"))
        if self.input_shape[:3] != (1, 2, 3) or self.output_shape[:2] != (1, 2):
            raise ValueError(
                f"DA3 engine must have fixed two-view shapes, got "
                f"{self.input_shape} -> {self.output_shape}"
            )
        if self.output_shape[-2:] != self.input_shape[-2:]:
            raise ValueError("DA3 depth resolution must match its image input")

        self.height, self.width = self.input_shape[-2:]
        self.input = torch.empty(self.input_shape, dtype=torch.float32, device="cuda")
        self.depth = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")
        self.confidence = torch.empty_like(self.depth)
        self.stream = torch.cuda.Stream()
        for name, tensor in (
            ("images", self.input),
            ("depth", self.depth),
            ("confidence", self.confidence),
        ):
            self.context.set_tensor_address(name, tensor.data_ptr())

    def _prepare_image(self, bgr_image):
        resized = cv2.resize(
            bgr_image,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self._MEAN) / self._STD
        return normalized.transpose(2, 0, 1), rgb

    def __call__(self, images):
        if len(images) != 2:
            raise ValueError(f"DA3 two-view inference requires 2 images, got {len(images)}")
        prepared, resized_rgb = zip(*(self._prepare_image(image) for image in images))
        input_cpu = torch.from_numpy(np.ascontiguousarray(np.stack(prepared)[None]))

        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            self.input.copy_(input_cpu)
            success = self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not success:
            raise RuntimeError("DA3 TensorRT inference failed")

        return (
            self.depth[0].cpu().numpy().copy(),
            self.confidence[0].cpu().numpy().copy(),
            np.stack(resized_rgb),
        )


@dataclass
class DenseKeyframe:
    timestamp: int
    camera_points: np.ndarray
    colors: np.ndarray
    aligned_depth: np.ndarray | None
    scale: float
    median_relative_error: float
    correspondences: int


class DenseMapBuilder:
    """Fuse fixed two-view DA3 depth into the evolving DPVO coordinate frame."""

    def __init__(
        self,
        engine_path,
        patch_graph,
        image_height,
        image_width,
        keyframe_delay=4,
        point_stride=7,
        max_alignment_error=0.25,
    ):
        if point_stride < 1:
            raise ValueError("dense-map point stride must be positive")
        if max_alignment_error <= 0:
            raise ValueError("dense-map maximum alignment error must be positive")
        self.inference = DA3TensorRT(engine_path)
        self.patch_graph = patch_graph
        self.image_height = image_height
        self.image_width = image_width
        self.keyframe_delay = keyframe_delay
        self.point_stride = point_stride
        self.max_alignment_error = max_alignment_error
        self.images = {}
        self.frames = {}
        self.rejected_pairs = []
        self.last_safe_timestamp = None

    def cache_image(self, timestamp, bgr_image):
        self.images[int(timestamp)] = np.ascontiguousarray(bgr_image.copy())

    def _index_for_timestamp(self, timestamp, num_frames):
        timestamps = self.patch_graph.tstamps_[:num_frames]
        indices = np.flatnonzero(timestamps == timestamp)
        return int(indices[0]) if len(indices) else None

    def _patch_samples(self, frame_index, depth, confidence):
        patch_size = self.patch_graph.patches_.shape[-1]
        center = patch_size // 2
        patches = (
            self.patch_graph.patches_[frame_index, :, :, center, center]
            .detach()
            .cpu()
            .numpy()
        )
        pixel_xy = 4.0 * (patches[:, :2] + 0.5)
        model_x = np.rint(
            (pixel_xy[:, 0] + 0.5) * depth.shape[1] / self.image_width - 0.5
        ).astype(np.int64)
        model_y = np.rint(
            (pixel_xy[:, 1] + 0.5) * depth.shape[0] / self.image_height - 0.5
        ).astype(np.int64)
        disparity = patches[:, 2]
        valid = (
            np.isfinite(patches).all(axis=1)
            & (disparity > 1e-4)
            & (model_x >= 0)
            & (model_x < depth.shape[1])
            & (model_y >= 0)
            & (model_y < depth.shape[0])
        )
        model_depth = depth[model_y[valid], model_x[valid]]
        model_confidence = confidence[model_y[valid], model_x[valid]]
        dpvo_depth = 1.0 / disparity[valid]
        finite = (
            np.isfinite(model_depth)
            & np.isfinite(model_confidence)
            & np.isfinite(dpvo_depth)
            & (model_depth > 1e-4)
            & (dpvo_depth > 1e-4)
        )
        model_depth = model_depth[finite]
        model_confidence = model_confidence[finite]
        dpvo_depth = dpvo_depth[finite]
        if len(model_depth) >= 8:
            confidence_cutoff = np.quantile(model_confidence, 0.25)
            confident = model_confidence >= confidence_cutoff
            model_depth = model_depth[confident]
            dpvo_depth = dpvo_depth[confident]
        return model_depth, dpvo_depth

    @staticmethod
    def _estimate_scale(model_depth, dpvo_depth):
        if len(model_depth) < 8:
            raise RuntimeError(
                f"not enough DA3/DPVO depth correspondences: {len(model_depth)}"
            )
        ratios = dpvo_depth / model_depth
        log_ratios = np.log(ratios)
        center = np.median(log_ratios)
        mad = np.median(np.abs(log_ratios - center))
        cutoff = max(3.0 * 1.4826 * mad, 0.10)
        inliers = np.abs(log_ratios - center) <= cutoff
        if inliers.sum() < 8:
            inliers = np.ones_like(ratios, dtype=bool)
        scale = float(np.median(ratios[inliers]))
        relative_error = np.abs(
            scale * model_depth[inliers] - dpvo_depth[inliers]
        ) / dpvo_depth[inliers]
        return scale, float(np.median(relative_error)), int(inliers.sum())

    def _make_keyframe(
        self,
        frame_index,
        timestamp,
        depth,
        confidence,
        rgb_image,
        scale,
        relative_error,
        correspondences,
    ):
        aligned_depth = depth * scale
        rows = np.arange(0, depth.shape[0], self.point_stride)
        cols = np.arange(0, depth.shape[1], self.point_stride)
        grid_y, grid_x = np.meshgrid(rows, cols, indexing="ij")
        sampled_depth = aligned_depth[grid_y, grid_x]
        sampled_confidence = confidence[grid_y, grid_x]
        valid = np.isfinite(sampled_depth) & (sampled_depth > 1e-4)
        if valid.any():
            valid &= sampled_confidence >= np.quantile(sampled_confidence[valid], 0.20)
            low, high = np.quantile(sampled_depth[valid], [0.01, 0.99])
            valid &= (sampled_depth >= low) & (sampled_depth <= high)

        pixel_x = (grid_x + 0.5) * self.image_width / depth.shape[1] - 0.5
        pixel_y = (grid_y + 0.5) * self.image_height / depth.shape[0] - 0.5
        intrinsics = (
            self.patch_graph.intrinsics_[frame_index].detach().cpu().numpy() * 4.0
        )
        x = (pixel_x - intrinsics[2]) / intrinsics[0] * sampled_depth
        y = (pixel_y - intrinsics[3]) / intrinsics[1] * sampled_depth
        camera_points = np.stack((x, y, sampled_depth), axis=-1)[valid].astype(np.float32)
        colors = rgb_image[grid_y, grid_x][valid].astype(np.uint8)
        return DenseKeyframe(
            timestamp=timestamp,
            camera_points=camera_points,
            colors=colors,
            aligned_depth=aligned_depth.astype(np.float32),
            scale=scale,
            median_relative_error=relative_error,
            correspondences=correspondences,
        )

    def _process_pair(self, previous_index, current_index):
        timestamps = self.patch_graph.tstamps_
        previous_timestamp = int(timestamps[previous_index])
        current_timestamp = int(timestamps[current_index])
        if previous_timestamp not in self.images or current_timestamp not in self.images:
            return [], False

        depth, confidence, rgb = self.inference(
            [self.images[previous_timestamp], self.images[current_timestamp]]
        )
        model_samples, dpvo_samples = [], []
        for index, predicted_depth, predicted_confidence in zip(
            (previous_index, current_index), depth, confidence
        ):
            model_depth, dpvo_depth = self._patch_samples(
                index,
                predicted_depth,
                predicted_confidence,
            )
            model_samples.append(model_depth)
            dpvo_samples.append(dpvo_depth)
        scale, relative_error, correspondences = self._estimate_scale(
            np.concatenate(model_samples),
            np.concatenate(dpvo_samples),
        )
        if relative_error > self.max_alignment_error:
            self.rejected_pairs.append(
                {
                    "previous_timestamp": previous_timestamp,
                    "current_timestamp": current_timestamp,
                    "scale": scale,
                    "median_relative_error": relative_error,
                    "correspondences": correspondences,
                }
            )
            return [], True

        updates = []
        for index, timestamp, predicted_depth, predicted_confidence, image in zip(
            (previous_index, current_index),
            (previous_timestamp, current_timestamp),
            depth,
            confidence,
            rgb,
        ):
            if timestamp in self.frames:
                continue
            frame = self._make_keyframe(
                index,
                timestamp,
                predicted_depth,
                predicted_confidence,
                image,
                scale,
                relative_error,
                correspondences,
            )
            self.frames[timestamp] = frame
            updates.append(frame)
        return updates, True

    def update(self, num_frames):
        safe_index = num_frames - self.keyframe_delay
        if safe_index < 1:
            return []
        safe_timestamp = int(self.patch_graph.tstamps_[safe_index])
        if safe_timestamp == self.last_safe_timestamp:
            return []

        if self.last_safe_timestamp is None:
            previous_index = safe_index - 1
        else:
            previous_index = self._index_for_timestamp(
                self.last_safe_timestamp,
                num_frames,
            )
            if previous_index is None:
                raise RuntimeError(
                    f"DPVO keyframe {self.last_safe_timestamp} disappeared after stabilization"
                )
        updates, processed = self._process_pair(previous_index, safe_index)
        if processed:
            self.last_safe_timestamp = safe_timestamp
            self.images = {
                timestamp: image
                for timestamp, image in self.images.items()
                if timestamp >= safe_timestamp
            }
        return updates

    def finalize(self, num_frames):
        if self.last_safe_timestamp is None:
            return []
        pending_indices = [
            index
            for index, timestamp in enumerate(self.patch_graph.tstamps_[:num_frames])
            if int(timestamp) > self.last_safe_timestamp and int(timestamp) in self.images
        ]
        updates = []
        for current_index in pending_indices:
            previous_index = self._index_for_timestamp(
                self.last_safe_timestamp,
                num_frames,
            )
            if previous_index is None:
                break
            pair_updates, processed = self._process_pair(previous_index, current_index)
            if not processed:
                break
            updates.extend(pair_updates)
            self.last_safe_timestamp = int(self.patch_graph.tstamps_[current_index])
        return updates

    def pose(self, timestamp, num_frames):
        index = self._index_for_timestamp(timestamp, num_frames)
        if index is None:
            return None
        return self.patch_graph.poses_[index].detach().cpu().numpy()

    def save(self, output_path, num_frames):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        world_points, colors = [], []
        frame_stats = []
        for timestamp, frame in sorted(self.frames.items()):
            pose = self.pose(timestamp, num_frames)
            if pose is None:
                continue
            rotation, translation = _camera_to_world(pose)
            world_points.append(frame.camera_points @ rotation.T + translation)
            colors.append(frame.colors)
            frame_stats.append(
                {
                    "timestamp": timestamp,
                    "points": len(frame.camera_points),
                    "scale": frame.scale,
                    "median_relative_error": frame.median_relative_error,
                    "correspondences": frame.correspondences,
                }
            )

        if not world_points:
            raise RuntimeError("dense map contains no aligned keyframes")
        world_points = np.concatenate(world_points).astype(np.float32)
        colors = np.concatenate(colors).astype(np.uint8)
        vertices = np.empty(
            len(world_points),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        vertices["x"], vertices["y"], vertices["z"] = world_points.T
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(output_path)

        metadata = {
            "coordinate_frame": "DPVO world (monocular scale)",
            "model": "Depth Anything 3 Small, fixed two-view TensorRT",
            "keyframes": len(frame_stats),
            "points": len(world_points),
            "point_stride": self.point_stride,
            "max_alignment_error": self.max_alignment_error,
            "rejected_pairs": self.rejected_pairs,
            "frames": frame_stats,
        }
        output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

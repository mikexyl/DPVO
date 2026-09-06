"""Camera-centered spherical snapshots of a bounded, sliding DA3 keyframe map."""

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .da3_dense import _camera_to_world


def project_points(points, colors, source_timestamps, width, region_ids=None,
                   region_colors=None, splat_radius=1):
    """Nearest radial surface per spherical pixel; RDF camera axes, rear seam.

    A bounded 3x3 angular splat covers small gaps between sampled depth points.
    It does not fill the remaining unobserved directions. Ties favor newer views.
    """
    height = width // 2
    ranges = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(ranges) & (ranges > 1e-4)
    points, ranges, colors, source_timestamps = (
        array[valid] for array in (points, ranges, colors, source_timestamps)
    )
    if region_ids is not None:
        region_ids, region_colors = region_ids[valid], region_colors[valid]
    u = np.floor((np.arctan2(points[:, 0], points[:, 2]) / (2 * np.pi) + 0.5) * width).astype(int) % width
    v = np.clip(np.floor((np.arcsin(np.clip(points[:, 1] / ranges, -1, 1)) / np.pi + 0.5)
                         * height).astype(int), 0, height - 1)
    # Reduce repeated views of the same angular pixel before expanding splats.
    # The radial minimum and timestamp tie-break commute with a fixed footprint,
    # so this gives the same z-buffer while bounding large-window intermediates.
    center_pixels = v * width + u
    order = np.lexsort((-source_timestamps.astype(np.int64), ranges, center_pixels))
    ordered_pixels = center_pixels[order]
    first = np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]] if len(order) else np.empty(0, bool)
    keep = order[first]
    u, v, ranges, colors, source_timestamps = (
        array[keep] for array in (u, v, ranges, colors, source_timestamps)
    )
    if region_ids is not None:
        region_ids, region_colors = region_ids[keep], region_colors[keep]
    pixels, indices = [], []
    for dy in range(-splat_radius, splat_radius + 1):
        rows = v + dy
        keep = (rows >= 0) & (rows < height)
        for dx in range(-splat_radius, splat_radius + 1):
            pixels.append(rows[keep] * width + (u[keep] + dx) % width)
            indices.append(np.flatnonzero(keep))
    pixels, indices = np.concatenate(pixels), np.concatenate(indices)
    order = np.lexsort((-source_timestamps[indices].astype(np.int64), ranges[indices], pixels))
    pixels, indices = pixels[order], indices[order]
    first = np.r_[True, pixels[1:] != pixels[:-1]] if len(pixels) else np.empty(0, bool)
    pixels, indices = pixels[first], indices[first]
    rgb = np.zeros((height * width, 4), np.uint8)
    rgb[pixels, :3], rgb[pixels, 3] = colors[indices], 255
    depth = np.full(height * width, np.nan, np.float32)
    depth[pixels] = ranges[indices]
    sources = np.full(height * width, -1, np.int32)
    sources[pixels] = source_timestamps[indices]
    labels = np.zeros(height * width, np.uint16)
    sam = None
    if region_ids is not None:
        labels[pixels] = region_ids[indices]
        sam = np.zeros_like(rgb)
        sam[pixels, :3], sam[pixels, 3] = region_colors[indices], 255
        sam = sam.reshape(height, width, 4)
    return dict(rgb=rgb.reshape(height, width, 4), sam=sam,
                radial_depth=depth.reshape(height, width),
                source_timestamp=sources.reshape(height, width),
                region_ids=labels.reshape(height, width))


def observed_sphere_mesh(coverage, radius):
    """Only tessellate fully observed 2x2 texel blocks; do not rely on texture alpha.

    Rerun mesh textures currently ignore alpha. Missing regions therefore have
    no triangles at all. UV origin is top-left, matching the equirectangular PNG.
    """
    height, width = coverage.shape
    rows, cols = height // 2, width // 2
    observed = coverage.reshape(rows, 2, cols, 2).all(axis=(1, 3))
    y, x = np.nonzero(observed)
    a = y * (cols + 1) + x
    b, c, d = a + 1, a + cols + 1, a + cols + 2
    triangles = np.stack((np.stack((a, c, b), axis=1), np.stack((b, c, d), axis=1)), axis=1).reshape(-1, 3)
    used, remapped = np.unique(triangles, return_inverse=True)
    uv = np.column_stack(((used % (cols + 1)) / cols, (used // (cols + 1)) / rows)).astype(np.float32)
    longitude, latitude = (uv[:, 0] - 0.5) * (2 * np.pi), (uv[:, 1] - 0.5) * np.pi
    vertices = radius * np.column_stack((np.cos(latitude) * np.sin(longitude),
                                         np.sin(latitude), np.cos(latitude) * np.cos(longitude)))
    return vertices.astype(np.float32), remapped.reshape(-1, 3).astype(np.uint32), uv


@dataclass
class LocalSphere:
    anchor_timestamp: int
    center: np.ndarray
    rotation: np.ndarray
    radius: float
    rgb: np.ndarray
    sam: np.ndarray | None
    radial_depth: np.ndarray
    source_timestamp: np.ndarray
    region_ids: np.ndarray
    source_centers: np.ndarray
    anchor_rgb: np.ndarray
    anchor_radial_depth: np.ndarray
    history_gain: np.ndarray
    metadata: dict


class LocalSphereBuilder:
    """Use the last W accepted keyframes, render every K new accepted keyframes."""

    def __init__(self, dense_map, output_dir, window_size=30, every=5, width=1024, radius=None):
        if not 2 <= window_size <= 60 or every < 1:
            raise ValueError("local sphere requires 2-60 window keyframes and a positive render interval")
        if not 256 <= width <= 2048 or width % 4:
            raise ValueError("local sphere width must be divisible by four, between 256 and 2048")
        if radius is not None and (not np.isfinite(radius) or radius <= 0):
            raise ValueError("sphere display radius must be finite and positive")
        self.dense_map = dense_map
        self.window = deque(maxlen=window_size)
        self.every, self.width, self.radius = every, width, radius
        self.accepted = 0
        self.last_render_count = 0
        self.last_timestamp = None
        self.records = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "manifest.json").exists():
            raise FileExistsError(f"use a new local-sphere output directory: {self.output_dir}")

    def ingest(self, frames, num_frames):
        results = []
        for frame in frames:
            if self.last_timestamp is not None and frame.timestamp <= self.last_timestamp:
                continue
            self.last_timestamp = frame.timestamp
            self.window.append(frame)
            self.accepted += 1
            if len(self.window) == self.window.maxlen and (
                self.last_render_count == 0 or self.accepted - self.last_render_count >= self.every
            ):
                result = self._render(num_frames, final=False)
                if result is not None:
                    results.append(result)
        return results

    def finalize(self, num_frames):
        if self.accepted == self.last_render_count or len(self.window) < 2:
            return []
        result = self._render(num_frames, final=True)
        return [] if result is None else [result]

    def _render(self, num_frames, final):
        start = perf_counter()
        anchor = self.window[-1]
        pose = self.dense_map.pose(anchor.timestamp, num_frames)
        if pose is None:
            return None
        rotation, center = _camera_to_world(pose)
        points, colors, timestamps, ids, label_colors, transforms = [], [], [], [], [], []
        has_labels = any(frame.region_ids is not None for frame in self.window)
        sources = []
        for frame in self.window:
            pose = self.dense_map.pose(frame.timestamp, num_frames)
            if pose is None:
                continue
            source_rotation, source_center = _camera_to_world(pose)
            # Read the current optimized poses each time, never cache world points.
            points.append((frame.camera_points @ source_rotation.T + source_center - center) @ rotation)
            colors.append(frame.colors)
            timestamps.append(np.full(len(frame.camera_points), frame.timestamp, np.int32))
            if has_labels:
                ids.append(frame.region_ids if frame.region_ids is not None else
                           np.zeros(len(frame.camera_points), np.uint16))
                label_colors.append(frame.region_colors if frame.region_colors is not None else
                                    np.full_like(frame.colors, 96))
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3], transform[:3, 3] = source_rotation, source_center
            transforms.append(transform)
            sources.append(frame.timestamp)
        if len(points) < 2:
            return None
        point_count = sum(len(p) for p in points)
        atlas = project_points(np.concatenate(points), np.concatenate(colors), np.concatenate(timestamps),
                               self.width, np.concatenate(ids) if has_labels else None,
                               np.concatenate(label_colors) if has_labels else None)
        valid = np.isfinite(atlas["radial_depth"])
        if not valid.any():
            return None
        # Compare the accumulated local map against exactly the same anchor's
        # points, angular grid, depth filtering and splat settings.
        anchor_atlas = project_points(points[-1], colors[-1], timestamps[-1], self.width)
        anchor_valid = np.isfinite(anchor_atlas['radial_depth'])
        added = valid & ~anchor_valid
        history_gain = atlas['rgb'].copy()
        history_gain[~added] = 0
        radius = self.radius if self.radius is not None else float(0.2 * np.median(atlas["radial_depth"][valid]))
        # Solid-angle coverage accounts for equirectangular polar distortion.
        weights = np.cos(((np.arange(self.width // 2) + 0.5) / (self.width // 2) - 0.5) * np.pi)[:, None]
        metadata = dict(anchor_timestamp=anchor.timestamp, source_timestamps=sources,
                        accepted_count=self.accepted, final=final, points=point_count,
                        center=center.tolist(), camera_to_world_rotation=rotation.tolist(),
                        source_camera_to_world=np.asarray(transforms).tolist(), display_radius=radius,
                        width=self.width, height=self.width // 2,
                        pixel_coverage=float(valid.mean()),
                        solid_angle_coverage=float((valid * weights).sum() / (weights.sum() * self.width)),
                        anchor_solid_angle_coverage=float((anchor_valid * weights).sum() / (weights.sum() * self.width)),
                        history_added_solid_angle=float((added * weights).sum() / (weights.sum() * self.width)),
                        history_only_fraction=float((added * weights).sum() / (valid * weights).sum()),
                        source_timestamp_span=int(sources[-1] - sources[0]),
                        projection_seconds=perf_counter() - start)
        sphere = LocalSphere(anchor.timestamp, center, rotation, radius,
                             source_centers=(np.asarray(transforms)[:, :3, 3] - center) @ rotation,
                             anchor_rgb=anchor_atlas['rgb'], anchor_radial_depth=anchor_atlas['radial_depth'],
                             history_gain=history_gain,
                             metadata=metadata, **atlas)
        self._save(sphere)
        self.last_render_count = self.accepted
        return sphere

    def _save(self, sphere):
        start = perf_counter()
        directory = self.output_dir / f"keyframe_{sphere.anchor_timestamp:06d}"
        directory.mkdir(exist_ok=False)
        for name, image in (("rgb", sphere.rgb), ("sam", sphere.sam),
                            ("anchor_rgb", sphere.anchor_rgb), ("history_gain", sphere.history_gain)):
            if image is not None:
                if not cv2.imwrite(str(directory / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)):
                    raise OSError(f"could not write {name} sphere image")
        np.savez_compressed(directory / "projection.npz", radial_depth=sphere.radial_depth,
                            source_timestamp=sphere.source_timestamp, region_ids=sphere.region_ids,
                            anchor_radial_depth=sphere.anchor_radial_depth,
                            center=sphere.center, rotation=sphere.rotation, display_radius=sphere.radius)
        sphere.metadata["save_seconds"] = perf_counter() - start
        (directory / "metadata.json").write_text(json.dumps(sphere.metadata, indent=2) + "\n")
        self.records.append(dict(sphere.metadata, directory=directory.name))
        manifest = dict(mode="sliding_local_map_at_newest_keyframe_camera", window_size=self.window.maxlen,
                        every=self.every, width=self.width, display_radius=self.radius or "0.2 * median radial depth",
                        coordinates="DPVO monocular scale; camera RDF; longitude atan2(x,z), latitude asin(y/r)",
                        visibility="nearest radial point with a 3x3 angular splat; unseen pixels transparent/NaN",
                        region_semantics="SAM region IDs, not semantic classes or global 3D identities",
                        snapshots=self.records)
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

"""Offline virtual spheres: source RGB -> geodesic samples, with depth visibility.

The saved panorama texture/depth raster is never an extraction input. This
prototype recovers filtered depth-grid topology from the grouped dense PLY and
final trajectory, then applies each snapshot's original source/anchor poses.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from plyfile import PlyData

from .da3_dense import _quaternion_to_matrix
from .sphere_bow import SphereFeatures
from .sphorb import SphorbExtractor

FAMILY = 'sphorb-virtual'
VERSION = 'e5f2ccf-source-grid-v1'
CELLS = (256, 204, 162, 128, 102, 80, 64)


def depth_mesh(camera_points, grid_xy, grid_size=(252, 189), max_jump=.05):
    """Connect only fully observed 2x2 cells with bounded relative depth jumps."""
    if not 0 < max_jump < 1:
        raise ValueError('mesh depth jump must be in (0,1)')
    xy = np.asarray(grid_xy)
    if xy.shape != (len(camera_points), 2) or not np.isfinite(camera_points).all() or (camera_points[:, 2] <= 0).any():
        raise ValueError('invalid source camera points/grid indices')
    if not np.issubdtype(xy.dtype, np.integer) or (xy < 0).any() or (xy >= grid_size).any():
        raise ValueError('invalid depth grid indices')
    indices = np.full((grid_size[1], grid_size[0]), -1, np.int32)
    if len(np.unique(xy[:, 1] * grid_size[0] + xy[:, 0])) != len(xy):
        raise ValueError('duplicate depth grid address')
    indices[xy[:, 1], xy[:, 0]] = np.arange(len(xy))
    corners = np.stack((indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]), axis=-1)
    corners = corners[(corners >= 0).all(axis=-1)]
    z = camera_points[corners, 2]
    corners = corners[np.max(z, axis=1) <= (1 + max_jump) * np.min(z, axis=1)]
    return np.concatenate((corners[:, [0, 2, 1]], corners[:, [1, 2, 3]])).astype(np.int32)


class SavedKeyframes:
    """Saved geometry plus original video frames or a DPVO image-stream directory."""
    def __init__(self, dense_map, trajectory, video, calibration, cache, stride=5, skip=0, max_jump=.05):
        self.dense_map, self.video, self.cache = Path(dense_map).resolve(), Path(video).resolve(), Path(cache).resolve()
        self.metadata = json.loads(self.dense_map.with_suffix('.json').read_text())
        self.vertices = PlyData.read(self.dense_map, mmap='r')['vertex'].data
        self.poses = np.loadtxt(trajectory)
        calib = np.loadtxt(calibration)
        if calib.shape != (4,) or stride < 1 or skip < 0 or (calib[:2] <= 0).any():
            raise ValueError('prototype requires four pinhole intrinsics, positive stride, nonnegative skip')
        self.is_images = self.video.is_dir()
        self.input_scale = 1. if self.is_images else .5
        self.intrinsics = calib * self.input_scale
        self.stride, self.skip, self.max_jump = stride, skip, max_jump
        if self.is_images:
            self.image_files = sorted(p for extension in ('*.png','*.jpeg','*.jpg') for p in self.video.glob(extension))
            if not self.image_files:
                raise ValueError(f'no source images in {self.video}')
            first = cv2.imread(str(self.image_files[0]))
            if first is None:
                raise ValueError(f'cannot decode {self.image_files[0]}')
            self.raw_size = first.shape[1::-1]
        else:
            cap = cv2.VideoCapture(str(self.video))
            if not cap.isOpened():
                raise ValueError(f'cannot open {self.video}')
            self.raw_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
        self.processed_size = tuple(int(s * self.input_scale) // 16 * 16 for s in self.raw_size)
        self.model_size = (504, 378)
        self.point_stride = self.metadata['point_stride']
        self.frames, offset = {}, 0
        for frame in self.metadata['frames']:
            timestamp = frame['timestamp']
            if timestamp in self.frames or timestamp >= len(self.poses) or self.poses[timestamp, 0] != timestamp:
                raise ValueError('invalid frame grouping or trajectory timestamps')
            self.frames[timestamp] = (offset, offset + frame['points'])
            offset += frame['points']
        if offset != len(self.vertices):
            raise ValueError('dense PLY grouping does not match metadata')
        fingerprint = dict(video=str(self.video), video_size=self.video.stat().st_size,
                           video_mtime=self.video.stat().st_mtime_ns, stride=stride, skip=skip,
                           trajectory_sha256=hashlib.sha256(Path(trajectory).read_bytes()).hexdigest(),
                           dense_metadata_sha256=hashlib.sha256(self.dense_map.with_suffix('.json').read_bytes()).hexdigest(),
                           calibration=calib.tolist(), raw_size=list(self.raw_size),
                           frame_mapping='raw zero-based frame = skip + (timestamp+1)*stride - 1')
        if self.is_images:
            # Preserve the old video fingerprint so validated Office caches remain usable.
            fingerprint.pop('video');fingerprint.pop('video_size');fingerprint.pop('video_mtime')
            listing=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in self.image_files]
            fingerprint.update(source_kind='images',images=str(self.video),image_count=len(listing),
                image_listing_sha256=hashlib.sha256(json.dumps(listing).encode()).hexdigest(),
                input_resize_scale=1.,processed_size=list(self.processed_size),
                frame_mapping='sorted raw image index = skip + timestamp*stride')
        self.cache.mkdir(parents=True, exist_ok=True)
        manifest = self.cache / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
            raise ValueError('source image cache provenance mismatch')
        manifest.write_text(json.dumps(fingerprint, indent=2) + '\n')
        self.fingerprint = fingerprint

    @lru_cache(maxsize=60)
    def geometry(self, timestamp):
        start, end = self.frames[timestamp]
        data = self.vertices[start:end]
        world = np.column_stack([data[k] for k in ('x', 'y', 'z')])
        pose = self.poses[timestamp]
        camera = ((world - pose[1:4]) @ _quaternion_to_matrix(pose[4:])).astype(np.float32)
        uv = camera[:, :2] / camera[:, 2, None] * self.intrinsics[:2] + self.intrinsics[2:]
        grid = (uv + .5) * (np.array(self.model_size) / self.processed_size) - .5
        xy = np.rint(grid / self.point_stride).astype(np.int32)
        error = np.max(abs(grid - xy * self.point_stride), initial=0)
        # Float32 world-point export introduces small errors when recovering
        # camera coordinates. This remains far below the stride-2 ambiguity
        # boundary; decoded point colors are also checked exactly below.
        if error > .05:
            raise ValueError(f'cannot recover exact source depth grid at {timestamp}: {error} pixels')
        triangles = depth_mesh(camera, xy, tuple((s + self.point_stride - 1) // self.point_stride for s in self.model_size), self.max_jump)
        # Invert the actual stream resize with pixel centers. Image directories
        # have scale 1, whereas DPVO video input has scale .5 before cropping.
        source_uv = ((xy * self.point_stride + .5) * (np.array(self.processed_size) / self.model_size)
                     / self.input_scale - .5).astype(np.float32)
        return camera, triangles, source_uv, xy, float(error)

    def raw_index(self, timestamp):
        return self.skip + timestamp*self.stride if self.is_images else self.skip+(timestamp+1)*self.stride-1

    def processed_image(self, image):
        if image is None or image.shape[1::-1] != self.raw_size:
            raise ValueError('missing source image or inconsistent original dimensions')
        if not self.is_images:
            image = cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        return image[:self.processed_size[1], :self.processed_size[0]]

    def original_images(self, timestamps):
        if self.is_images:
            for timestamp in timestamps:
                index=self.raw_index(timestamp)
                if not 0 <= index < len(self.image_files):
                    raise ValueError(f'image sequence ended before source frame {index}')
                yield timestamp,index,cv2.imread(str(self.image_files[index]))
        else:
            targets={self.raw_index(t):t for t in timestamps}
            cap=cv2.VideoCapture(str(self.video))
            try:
                for index in range(max(targets)+1):
                    if not cap.grab():
                        raise ValueError(f'video ended before source frame {index}')
                    if index in targets:
                        ok,image=cap.retrieve()
                        if not ok:raise ValueError('could not decode source frame')
                        yield targets[index],index,image
            finally:
                cap.release()

    def prepare(self, timestamps):
        missing = [t for t in sorted(set(timestamps)) if not all((self.cache/f'{t:06d}.{ext}').exists() for ext in ('png','json'))]
        if not missing:
            return
        done = 0
        for timestamp,frame_index,image in self.original_images(missing):
            _, _, _, xy, error = self.geometry(timestamp)
            processed = self.processed_image(image)
            rgb = cv2.cvtColor(cv2.resize(processed, self.model_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            start, end = self.frames[timestamp]
            data = self.vertices[start:end]
            expected = np.column_stack([data[k] for k in ('red', 'green', 'blue')])
            actual = rgb[xy[:, 1] * self.point_stride, xy[:, 0] * self.point_stride]
            if not np.array_equal(actual, expected):
                raise ValueError(f'source frame/color mismatch at {timestamp}; check stride/skip/calibration')
            if not cv2.imwrite(str(self.cache / f'{timestamp:06d}.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError('could not cache original image')
            record=dict(timestamp=timestamp,raw_frame=frame_index,source_color_exact=True,points=len(actual),grid_error_max=error)
            if self.is_images:
                record.update(source_image=str(self.image_files[frame_index]),
                    source_sha256=hashlib.sha256(self.image_files[frame_index].read_bytes()).hexdigest())
            (self.cache / f'{timestamp:06d}.json').write_text(json.dumps(record) + '\n')
            done += 1
            if done % 20 == 0 or done == len(missing):
                print(f'Validated/cached original images {done}/{len(missing)}', flush=True)

    @lru_cache(maxsize=35)
    def pyramid(self, timestamp):
        image = cv2.imread(str(self.cache / f'{timestamp:06d}.png'))
        if image is None or image.shape[1::-1] != self.raw_size:
            raise ValueError('missing or invalid original keyframe image')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = [gray]
        while min(result[-1].shape) >= 16:
            h, w = result[-1].shape
            result.append(cv2.resize(result[-1], (w // 2, h // 2), interpolation=cv2.INTER_AREA))
        return result

    def scene(self, snapshot, depth_band=.03):
        from . import _sphorb
        vertices, triangles, uv, source_z, source_ids = [], [], [], [], []
        offset = 0
        rotation = np.asarray(snapshot['camera_to_world_rotation'], np.float64)
        center = np.asarray(snapshot['center'], np.float64)
        timestamps = snapshot['source_timestamps']
        if timestamps != sorted(set(timestamps)) or len(timestamps) > 64:
            raise ValueError('source timestamps must be unique and chronological')
        transforms = snapshot['source_camera_to_world']
        if len(transforms) != len(timestamps):
            raise ValueError('source pose count mismatch')
        for index, (timestamp, transform) in enumerate(zip(timestamps, transforms)):
            camera, tri, coords, _, _ = self.geometry(timestamp)
            pose = np.asarray(transform)
            vertices.append(((camera @ pose[:3, :3].T + pose[:3, 3] - center) @ rotation).astype(np.float32))
            triangles.append(tri + offset)
            uv.append(coords)
            source_z.append(camera[:, 2].copy())
            source_ids.append(np.full(len(tri), index, np.int32))
            offset += len(camera)
        arrays = [np.ascontiguousarray(np.concatenate(a)) for a in (vertices, triangles, uv, source_z, source_ids)]
        return _sphorb.VirtualScene(*arrays, depth_band), dict(vertices=len(arrays[0]), triangles=len(arrays[1]))


def sample_texture(hit, pyramids, angular_step, source_offset=0):
    """Fetch filtered original pixels; preserve unknowns and fractional provenance."""
    count = len(hit['source'])
    gray = np.zeros(count, np.uint8)
    valid = np.zeros(count, np.uint8)
    levels = np.full(count, 255, np.uint8)
    for source, pyramid in enumerate(pyramids, start=source_offset):
        ids = np.flatnonzero(hit['source'] == source)
        if not len(ids):
            continue
        footprint = np.maximum(1, hit['pixel_scale'][ids] * np.asarray(angular_step)) if np.ndim(angular_step) == 0 else np.maximum(1, hit['pixel_scale'][ids] * angular_step[ids])
        lod = np.floor(np.log2(footprint)).clip(0, len(pyramid) - 1).astype(int)
        for level in np.unique(lod):
            selected = ids[lod == level]
            image = pyramid[level]
            # Actual dimensions handle odd-sized mip levels exactly.
            scale = np.array(image.shape[::-1]) / pyramid[0].shape[::-1]
            uv = (hit['uv'][selected] + .5) * scale - .5
            supported = np.isfinite(uv).all(axis=1) & (uv >= 0).all(axis=1) & (uv < np.array(image.shape[::-1]) - 1).all(axis=1)
            selected, uv = selected[supported], uv[supported]
            if not len(selected):
                continue
            coords = uv.astype(np.float32)
            # OpenCV remap dimensions are limited to SHRT_MAX: sample in chunks.
            for start in range(0, len(selected), 30000):
                stop = start + 30000
                gray[selected[start:stop]] = cv2.remap(image, coords[start:stop, 0, None], coords[start:stop, 1, None], cv2.INTER_LINEAR).ravel()
            valid[selected] = 255
            levels[selected] = level
    return gray, valid, levels


class VirtualSphorbExtractor:
    def __init__(self, features=3000, levels=7, threshold=20, workers=4):
        self.extractor = SphorbExtractor(features, levels, threshold, workers)
        self.workers, self.levels = workers, levels
        self.candidates = SphorbExtractor(1000000, levels, threshold, workers)
        self.rays = [self.extractor.native.grid_bearings(l) for l in range(levels)]
        self.steps = []
        for rays in self.rays:
            dx = np.linalg.norm(rays[:, :, 1:] - rays[:, :, :-1], axis=-1)
            dy = np.linalg.norm(rays[:, 1:] - rays[:, :-1], axis=-1)
            step = np.maximum(np.pad(dx, ((0, 0), (0, 0), (0, 1)), mode='edge'),
                              np.pad(dy, ((0, 0), (0, 1), (0, 0)), mode='edge'))
            self.steps.append(step.ravel())

    def __call__(self, scene, pyramids, timestamps, display_size=(1024, 512), provenance_dir=None):
        from scipy.spatial import cKDTree
        visibility = []
        timing = dict(visibility_seconds=0., raytrace_seconds=0., texture_seconds=0., detector_native_seconds=0.,
                      provenance_write_seconds=0.)
        start = perf_counter()
        for rays in self.rays:
            visibility.append(scene.sample(rays.reshape(-1, 3), self.workers)['visible_sources'])
        timing['visibility_seconds'] = perf_counter() - start
        candidates, provenance = [], []
        for source, (timestamp, pyramid) in enumerate(zip(timestamps, pyramids)):
            gray, valid, source_grids, hits, addresses, lods = [], [], [], [], [], []
            for level, rays in enumerate(self.rays):
                selected = np.flatnonzero(visibility[level] & (np.uint64(1) << np.uint64(source)))
                start = perf_counter()
                hit = scene.sample(np.ascontiguousarray(rays.reshape(-1, 3)[selected]), self.workers, source)
                timing['raytrace_seconds'] += perf_counter() - start
                start = perf_counter()
                image, mask, lod = sample_texture(hit, [pyramid], self.steps[level][selected], source_offset=source)
                timing['texture_seconds'] += perf_counter() - start
                shape = rays.shape[:-1]
                g = np.zeros(shape, np.uint8); g.ravel()[selected] = image
                m = np.zeros(shape, np.uint8); m.ravel()[selected] = mask
                ids = np.where(m > 0, source + 1, 0).astype(np.uint8)
                gray.append(g); valid.append(m); source_grids.append(ids)
                hits.append(hit); addresses.append(selected); lods.append(lod)
                if provenance_dir is not None:
                    start = perf_counter()
                    provenance_dir.mkdir(parents=True, exist_ok=True)
                    keep = mask > 0
                    np.savez_compressed(provenance_dir / f'source_{timestamp:06d}_level_{level}.npz',
                                        shape=shape, indices=selected[keep].astype(np.int32), gray=image[keep],
                                        source_uv=hit['uv'][keep], depth=hit['depth'][keep], source_lod=lod[keep],
                                        triangle=hit['triangle'][keep], source_timestamp=timestamp)
                    timing['provenance_write_seconds'] += perf_counter() - start
            data = self.candidates.native.extract_grids(gray, valid, source_grids, *display_size)
            timing['detector_native_seconds'] += data.pop('native_seconds')
            grid, sections = data.pop('grid'), data.pop('sections')
            n = len(data['uv'])
            depth = np.full(n, np.nan, np.float32); uv = np.full((n, 2), np.nan, np.float32)
            source_lod = np.full(n, 255, np.uint8)
            for l in range(self.levels):
                ids = np.flatnonzero(data['octaves'] == l)
                x, y = np.rint(grid[ids]).astype(int).T - np.array([[17], [18]])
                location = (sections[ids] * (CELLS[l] + 1) + y) * (2 * CELLS[l] + 1) + x
                sampled = np.searchsorted(addresses[l], location)
                hit = hits[l]
                depth[ids] = hit['depth'][sampled]; uv[ids] = hit['uv'][sampled]
                source_lod[ids] = lods[l][sampled]
            data['points'] = data['bearings'] * depth[:, None]
            candidates.append(data)
            provenance.append(dict(source_timestamps=np.full(n, timestamp, np.int32), source_uv=uv,
                                   source_lod=source_lod, grid_xy=grid, grid_sections=sections, octaves=data['octaves']))
        combined = {key: np.concatenate([c[key] for c in candidates]) for key in candidates[0]}
        prov = {key: np.concatenate([p[key] for p in provenance]) for key in provenance[0]}
        keep = []
        for level, quota in enumerate(self.extractor.native.quotas()):
            ids = np.flatnonzero(combined['octaves'] == level)
            # Stable response sorting; ties favor the newer source, then its
            # deterministic native site order. Spatial suppression removes
            # duplicate observations before enforcing the published quotas.
            order = np.lexsort((ids, -prov['source_timestamps'][ids], -combined['responses'][ids]))
            tree = cKDTree(combined['bearings'][ids])
            suppressed = np.zeros(len(ids), bool)
            selected = []
            radius = float(np.median(self.steps[level])) * 1.5
            for index in order:
                if suppressed[index]:
                    continue
                selected.append(ids[index])
                suppressed[tree.query_ball_point(combined['bearings'][ids[index]], radius)] = True
                if len(selected) >= quota:
                    break
            keep.extend(selected)
        keep = np.asarray(keep, np.int32)
        result = SphereFeatures(**{k: v[keep] for k, v in combined.items()}, face_ids=None, face_xy=None,
                                descriptor_family=FAMILY, descriptor_version=VERSION)
        provenance = {key: value[keep] for key, value in prov.items()}
        timing['coherent_candidates'] = sum(len(c['uv']) for c in candidates)
        return result, provenance, timing

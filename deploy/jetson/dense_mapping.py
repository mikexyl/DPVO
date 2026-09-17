"""Bounded, asynchronous DA3 TensorRT mapping in DPVO's stable session frame."""
from collections import OrderedDict
import multiprocessing as mp
from queue import Empty
import time

import numpy as np


def sample_depth(image, uv):
    """Bilinear samples; out-of-image observations are invalid rather than clipped."""
    h, w = image.shape
    uv = np.asarray(uv)
    valid = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    xy = np.where(valid[:, None], uv, 0)
    x, y = np.floor(xy).astype(int).T
    dx, dy = (xy - np.floor(xy)).T
    result = ((1-dx)*(1-dy)*image[y, x] + dx*(1-dy)*image[y, x+1]
              + (1-dx)*dy*image[y+1, x] + dx*dy*image[y+1, x+1])
    return np.where(valid, result, np.nan)


def align_scale(depth, confidence, uv, target_depth, min_matches=16):
    """Robust multiplicative alignment to mature landmark camera-Z (not range)."""
    predicted = sample_depth(depth, uv)
    conf = sample_depth(confidence, uv)
    valid_conf = confidence[np.isfinite(confidence) & (confidence > 0)]
    if not len(valid_conf):
        raise ValueError('No valid depth confidence')
    threshold = np.percentile(valid_conf, 40)
    valid = np.isfinite(predicted) & (predicted > 0) & np.isfinite(target_depth)
    valid &= (target_depth > 0) & np.isfinite(conf) & (conf >= threshold)
    if valid.sum() < min_matches:
        raise ValueError('Too few confident landmark matches')
    ratios = np.log(target_depth[valid] / predicted[valid])
    median = np.median(ratios)
    mad = np.median(np.abs(ratios - median))
    inliers = np.abs(ratios - median) <= max(np.log(1.05), 3 * 1.4826 * mad)
    if inliers.sum() < min_matches or inliers.mean() < .6:
        raise ValueError('Inconsistent landmark depth scale')
    residual = float(np.median(np.abs(ratios[inliers] - np.median(ratios[inliers]))))
    if residual > .2:
        raise ValueError('DA3 depth disagrees with DPVO geometry')
    return float(np.exp(np.median(ratios[inliers]))), dict(
        matches=int(valid.sum()), inliers=int(inliers.sum()), log_residual=residual,
        confidence_threshold=float(threshold))


def stable_camera_pose(session_from_map, camera_to_map):
    """Separate Sim(3) scale from rigid camera pose; depths absorb that scale."""
    scale = float(np.cbrt(np.linalg.det(session_from_map[:3, :3])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid session gauge')
    pose = np.eye(4)
    pose[:3, :3] = session_from_map[:3, :3] @ camera_to_map[:3, :3] / scale
    pose[:3, 3] = session_from_map[:3, :3] @ camera_to_map[:3, 3] + session_from_map[:3, 3]
    return scale, pose


def backproject(depth, confidence, bgr, intrinsics, scale, threshold, stride=3):
    import cv2
    h, w = depth.shape
    rgb = cv2.resize(bgr[..., ::-1], (w, h), interpolation=cv2.INTER_LINEAR)
    y, x = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[y, x] * scale
    valid = np.isfinite(z) & (z > 0) & np.isfinite(confidence[y, x])
    valid &= confidence[y, x] >= threshold
    # Remove long depth discontinuities, where bilinear depth and color disagree.
    relative_dx = np.abs(np.diff(depth, axis=1, append=depth[:, -1:])) / np.maximum(depth, 1e-6)
    relative_dy = np.abs(np.diff(depth, axis=0, append=depth[-1:])) / np.maximum(depth, 1e-6)
    valid &= (relative_dx[y, x] < .1) & (relative_dy[y, x] < .1)
    fx, fy, cx, cy = intrinsics
    points = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], axis=-1)
    return points[valid].astype(np.float32), rgb[y, x][valid]


class RollingCloud:
    """Camera-local keyframe clouds, reposed with current BA and bounded in size."""
    def __init__(self, max_frames=20, max_points=50000, voxel_size=.02):
        if max_frames < 1 or max_points < 1 or not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError('Dense map limits must be positive')
        self.frames = OrderedDict()
        self.max_frames, self.max_points, self.voxel_size = max_frames, max_points, voxel_size

    def add(self, key, points, colors):
        self.frames[key] = (points, colors)
        self.frames.move_to_end(key)
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)

    def fused(self, poses):
        # Culled keyframes have no current BA pose; drop their clouds.
        for key in list(self.frames):
            if key not in poses:
                del self.frames[key]
        points, colors = [], []
        for key, (xyz, rgb) in self.frames.items():
            pose = poses[key]
            points.append(xyz @ pose[:3, :3].T + pose[:3, 3])
            colors.append(rgb)
        if not points:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
        points, colors = np.concatenate(points), np.concatenate(colors)
        finite = np.isfinite(points).all(axis=1)
        points, colors = points[finite], colors[finite]
        _, indices = np.unique(np.floor(points / self.voxel_size), axis=0, return_index=True)
        step = max(1, (len(indices) + self.max_points - 1) // self.max_points)
        indices = indices[::step]
        return points[indices].astype(np.float32), colors[indices]


def _worker(engine_path, jobs, results):
    # This is a spawned child: CUDA allocations vanish on Stop, never fork CUDA.
    try:
        from deploy.jetson.da3_runtime import Da3TensorRT
        from deploy.jetson.da3_export import preprocess
        engine = Da3TensorRT(engine_path)
        h, w = engine.shape[-2:]
        while True:
            job = jobs.get()
            if job is None:
                return
            key, bgr, intrinsics, uv, target = job
            start = time.monotonic()
            try:
                depth, confidence = engine(preprocess(bgr, h, w))
                depth, confidence = depth.reshape(h, w), confidence.reshape(h, w)
                ratio = np.array([w / bgr.shape[1], h / bgr.shape[0]])
                # Pixel-center resize convention must match RGB interpolation.
                uv = (uv + .5) * ratio - .5
                intrinsics = intrinsics.copy()
                intrinsics[:2] *= ratio
                intrinsics[2:] = (intrinsics[2:] + .5) * ratio - .5
                scale, stats = align_scale(depth, confidence, uv, target)
                xyz, rgb = backproject(depth, confidence, bgr, intrinsics, scale,
                                       stats['confidence_threshold'])
                stats.update(scale=scale, inference_ms=(time.monotonic() - start) * 1000)
                results.put((key, xyz, rgb, stats))
            except ValueError as error:
                results.put((key, None, None, {'rejected': str(error)}))
    except Exception as error:
        results.put((None, None, None, {'fatal': str(error)}))


class OnlineDenseMapper:
    def __init__(self, engine_path, fps=.5, max_points=50000, voxel_size=.02):
        from pathlib import Path
        if not np.isfinite(fps) or fps <= 0 or not Path(engine_path).is_file():
            raise ValueError('Dense mapping requires a validated TensorRT engine and positive FPS')
        context = mp.get_context('spawn')
        self.jobs, self.results = context.Queue(maxsize=1), context.Queue(maxsize=2)
        self.process = context.Process(target=_worker, args=(engine_path, self.jobs, self.results), daemon=True)
        self.process.start()
        self.cache = OrderedDict()
        self.cloud = RollingCloud(max_points=max_points, voxel_size=voxel_size)
        self.last_submit, self.interval = 0., 1 / fps
        self.pending = False
        self.last_key = None
        self.poses = {}
        self.stats = {'state': 'loading TensorRT'}

    def remember(self, key, bgr):
        self.cache[key] = bgr.copy()
        while len(self.cache) > 64:
            self.cache.popitem(last=False)

    def update(self, slam, intrinsics):
        from dpvo.lietorch import SE3
        changed = False
        try:
            key, xyz, rgb, self.stats = self.results.get_nowait()
            self.pending = False
            if 'fatal' in self.stats:
                raise RuntimeError(self.stats['fatal'])
            if xyz is not None:
                self.cloud.add(key, xyz, rgb)
                changed = True
        except Empty:
            pass
        if not self.process.is_alive():
            raise RuntimeError('DA3 TensorRT worker exited')
        if not slam.is_initialized or slam.n < 8:
            return None
        now = time.monotonic()
        due = not self.pending and now - self.last_submit >= self.interval
        if not changed and not due:
            return None
        pg = slam.pg
        keys = [int(k) for k in pg.tstamps_[:slam.n]]
        matrices = SE3(pg.poses_[:slam.n]).inv().matrix().detach().float().cpu().numpy()
        gauge = np.asarray(pg.session_from_map_)
        self.poses = {key: stable_camera_pose(gauge, matrix)[1] for key, matrix in zip(keys, matrices)}
        if due:
            self.last_submit = now
            slot = slam.n - 4
            key = keys[slot]
            if key != self.last_key and key in self.cache:
                patches = pg.patches_[slot].detach().float().cpu().numpy()
                c = patches.shape[-1] // 2
                uv = patches[:, :2, c, c] * slam.RES
                gauge_scale, _ = stable_camera_pose(gauge, matrices[slot])
                with np.errstate(divide='ignore', invalid='ignore'):
                    target = gauge_scale / patches[:, 2, c, c]
                self.jobs.put_nowait((key, self.cache[key], intrinsics.copy(), uv, target))
                self.pending, self.last_key, self.last_submit = True, key, now
        return self.cloud.fused(self.poses) if changed or self.cloud.frames else None

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        for queue in (self.jobs, self.results):
            queue.cancel_join_thread()
            queue.close()
        self.cache.clear()
        self.cloud.frames.clear()

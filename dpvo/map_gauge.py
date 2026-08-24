"""Utilities for keeping an externally visible DPVO map frame stable."""

import numpy as np
from scipy.spatial.transform import Rotation


def similarity_scale(transform):
    """Return the positive uniform scale of a 4x4 Sim(3) matrix."""

    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    return float(np.cbrt(abs(np.linalg.det(linear))))


def normalization_change(camera_to_map, disparity_scale):
    """Return old-map-from-new-map for ``PatchGraph.normalize``.

    DPVO divides disparity by ``disparity_scale``, scales pose translations by
    the same value, and selects camera zero as the new map origin.
    """

    camera_to_map = np.asarray(camera_to_map, dtype=np.float64).reshape(4, 4)
    disparity_scale = float(disparity_scale)
    if not np.isfinite(disparity_scale) or disparity_scale <= 0.0:
        raise ValueError("normalization scale must be finite and positive")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = camera_to_map[:3, :3] / disparity_scale
    output[:3, 3] = camera_to_map[:3, 3]
    return output


def transform_points(transform, points):
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points = np.asarray(points)
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_pose_xyzw(transform, pose):
    """Express a camera-to-map SE(3) pose in the stable session map."""

    return transform_poses_xyzw(transform, np.asarray(pose).reshape(1, 7))[0]


def transform_poses_xyzw(transform, poses):
    """Vectorized form of :func:`transform_pose_xyzw`."""

    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    scale = similarity_scale(transform)
    gauge_rotation = transform[:3, :3] / scale
    output = np.empty_like(poses)
    output[:, :3] = transform_points(transform, poses[:, :3])
    output[:, 3:] = Rotation.from_matrix(
        gauge_rotation[None] @ Rotation.from_quat(poses[:, 3:]).as_matrix()
    ).as_quat()
    return output


def scale_camera_points(transform, points):
    """Move camera-frame points into the stable session's length unit."""

    return np.asarray(points) * similarity_scale(transform)

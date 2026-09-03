"""Pure calibration and compressed-image helpers for S3E playback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cu_multi_core import PinholeCalibration


@dataclass(frozen=True)
class S3ECalibration:
    camera: PinholeCalibration
    fps: float
    t_imu_camera: np.ndarray


def _matrix(storage: cv2.FileStorage, key: str, shape: tuple[int, int]) -> np.ndarray:
    node = storage.getNode(key)
    if node.empty():
        raise KeyError(f"S3E calibration has no {key!r}")
    value = node.mat()
    if value is None:
        raise ValueError(f"S3E calibration {key!r} is not a matrix")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"S3E calibration {key!r} must be a finite {shape} matrix")
    return matrix


def _scalar(storage: cv2.FileStorage, key: str) -> float:
    node = storage.getNode(key)
    if node.empty():
        raise KeyError(f"S3E calibration has no {key!r}")
    value = float(node.real())
    if not np.isfinite(value):
        raise ValueError(f"S3E calibration {key!r} is not finite")
    return value


def read_s3e_calibration(path: Path) -> S3ECalibration:
    """Read one official S3E OpenCV YAML calibration."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"S3E calibration does not exist: {path}")
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"cannot open S3E calibration: {path}")
    try:
        camera_type = storage.getNode("Camera.type").string().lower()
        if camera_type != "pinhole":
            raise ValueError(
                f"S3E Camera.type must be PinHole, got {camera_type!r}"
            )
        width = int(_scalar(storage, "LEFT.width"))
        height = int(_scalar(storage, "LEFT.height"))
        camera_matrix = _matrix(storage, "LEFT.K", (3, 3))
        distortion = _matrix(storage, "LEFT.D", (1, 5)).reshape(-1)
        t_imu_camera = _matrix(storage, "Tic", (4, 4))
        fps = _scalar(storage, "Camera.fps")
    finally:
        storage.release()

    if width <= 0 or height <= 0 or fps <= 0.0:
        raise ValueError("S3E width, height, and FPS must be positive")
    if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("S3E LEFT.K has an invalid homogeneous row")
    if not np.allclose(t_imu_camera[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("S3E Tic has an invalid homogeneous row")
    rotation = t_imu_camera[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("S3E Tic rotation is not orthonormal")
    return S3ECalibration(
        camera=PinholeCalibration(
            width=width,
            height=height,
            camera_matrix=camera_matrix,
            distortion=distortion,
            distortion_model="plumb_bob",
        ),
        fps=fps,
        t_imu_camera=t_imu_camera,
    )


def decode_compressed_bgr8(data: bytes) -> np.ndarray:
    """Decode a ROS CompressedImage JPEG payload as contiguous BGR8."""

    encoded = np.frombuffer(bytes(data), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("compressed S3E image is empty")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("failed to decode compressed S3E image as BGR8 JPEG")
    return np.ascontiguousarray(image, dtype=np.uint8)

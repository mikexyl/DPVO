"""Pure helpers for calibrated Newer College cam0 playback.

This module deliberately has no ROS imports so image decoding, Kalibr parsing,
fisheye rectification, timestamp handling, and playback state can be tested in
an ordinary Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class NewerCollegeCameraCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    resolution: tuple[int, int]
    t_cam_imu: np.ndarray
    camera_model: str
    distortion_model: str


@dataclass(frozen=True)
class FisheyeRectification:
    map_x: np.ndarray
    map_y: np.ndarray
    camera_matrix: np.ndarray
    resolution: tuple[int, int]

    def rectify(self, image: np.ndarray) -> np.ndarray:
        expected_shape = (self.resolution[1], self.resolution[0])
        if image.ndim != 2 or image.shape != expected_shape:
            raise ValueError(
                f"cam0 image has shape {image.shape}, expected {expected_shape}"
            )
        output = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return np.ascontiguousarray(output, dtype=np.uint8)


def _matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    return matrix


def read_kalibr_camera_calibration(
    path: Path, camera_key: str = "cam0"
) -> NewerCollegeCameraCalibration:
    """Read the standard Kalibr camchain entry for Newer College cam0."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Newer College calibration does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or camera_key not in document:
        raise KeyError(f"calibration {path} has no {camera_key!r} entry")
    camera = document[camera_key]
    if not isinstance(camera, dict):
        raise ValueError(f"calibration {path} entry {camera_key!r} is not a mapping")

    camera_model = str(camera.get("camera_model", "")).lower()
    distortion_model = str(camera.get("distortion_model", "")).lower()
    if camera_model != "pinhole":
        raise ValueError(
            f"{camera_key} camera_model must be 'pinhole', got {camera_model!r}"
        )
    if distortion_model not in ("equidistant", "fisheye"):
        raise ValueError(
            f"{camera_key} distortion_model must be 'equidistant', "
            f"got {distortion_model!r}"
        )

    intrinsics = np.asarray(camera.get("intrinsics"), dtype=np.float64)
    distortion = np.asarray(camera.get("distortion_coeffs"), dtype=np.float64)
    resolution = np.asarray(camera.get("resolution"), dtype=np.int64)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise ValueError(f"{camera_key} intrinsics must be [fx, fy, cx, cy]")
    if distortion.shape != (4,) or not np.isfinite(distortion).all():
        raise ValueError(f"{camera_key} equidistant distortion must have 4 values")
    if resolution.shape != (2,) or np.any(resolution <= 0):
        raise ValueError(f"{camera_key} resolution must be [width, height]")
    t_cam_imu = _matrix(camera.get("T_cam_imu"), f"{camera_key}.T_cam_imu")
    camera_matrix = np.asarray(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return NewerCollegeCameraCalibration(
        camera_matrix=camera_matrix,
        distortion=distortion,
        resolution=(int(resolution[0]), int(resolution[1])),
        t_cam_imu=t_cam_imu,
        camera_model=camera_model,
        distortion_model="equidistant",
    )


def build_fisheye_rectification(
    calibration: NewerCollegeCameraCalibration,
    balance: float = 0.0,
) -> FisheyeRectification:
    """Precompute full-resolution OpenCV equidistant rectification maps."""

    if not 0.0 <= balance <= 1.0:
        raise ValueError("rectification balance must be in [0, 1]")
    width, height = calibration.resolution
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        calibration.camera_matrix,
        calibration.distortion,
        (width, height),
        np.eye(3, dtype=np.float64),
        balance=float(balance),
        new_size=(width, height),
    )
    if new_camera_matrix.shape != (3, 3) or not np.isfinite(
        new_camera_matrix
    ).all():
        raise RuntimeError("OpenCV produced invalid rectified cam0 intrinsics")
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.distortion,
        np.eye(3, dtype=np.float64),
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return FisheyeRectification(
        map_x=map_x,
        map_y=map_y,
        camera_matrix=new_camera_matrix,
        resolution=(width, height),
    )


def decode_compressed_mono8(data: bytes) -> np.ndarray:
    """Decode a ROS CompressedImage JPEG payload as contiguous mono8."""

    encoded = np.frombuffer(bytes(data), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("compressed cam0 image is empty")
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError("failed to decode compressed cam0 image as grayscale JPEG")
    return np.ascontiguousarray(image, dtype=np.uint8)


def stamp_components(stamp: Any) -> tuple[int, int]:
    """Return a normalized ROS header stamp without consulting bag time."""

    seconds = int(stamp.sec)
    if hasattr(stamp, "nanosec"):
        nanoseconds = int(stamp.nanosec)
    elif hasattr(stamp, "nsec"):
        nanoseconds = int(stamp.nsec)
    else:
        raise AttributeError("ROS stamp has neither nanosec nor nsec")
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError(f"invalid ROS timestamp {seconds}.{nanoseconds:09d}")
    return seconds, nanoseconds


def rectified_camera_info(rectification: FisheyeRectification) -> dict[str, Any]:
    """Return the distortion-free CameraInfo fields for a rectified image."""

    width, height = rectification.resolution
    matrix = rectification.camera_matrix
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    return {
        "width": width,
        "height": height,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "d": [],
        "distortion_model": "plumb_bob",
    }


class PlaybackState:
    """Stride/start/limit and acknowledgement state for a bag player."""

    def __init__(self, stride: int, start_frame: int, max_frames: int):
        if stride < 1:
            raise ValueError("stride must be positive")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        self.stride = int(stride)
        self.start_frame = int(start_frame)
        self.max_frames = int(max_frames)
        self.source_index = -1
        self.selected_index = -1
        self.processed = 0
        self.waiting_for_ack = False
        self.finished = False

    @property
    def limit_reached(self) -> bool:
        return self.max_frames > 0 and self.processed >= self.max_frames

    def select_next_source(self) -> bool:
        if self.waiting_for_ack:
            raise RuntimeError("cannot consume another source frame before acknowledgement")
        if self.finished or self.limit_reached:
            return False
        self.source_index += 1
        if self.source_index % self.stride:
            return False
        self.selected_index += 1
        return self.selected_index >= self.start_frame

    def mark_published(self) -> None:
        if self.waiting_for_ack or self.finished or self.limit_reached:
            raise RuntimeError("invalid playback publish transition")
        self.processed += 1
        self.waiting_for_ack = True

    def acknowledge(self) -> bool:
        if not self.waiting_for_ack:
            return False
        self.waiting_for_ack = False
        return self.limit_reached

    def finish(self) -> None:
        if self.waiting_for_ack:
            raise RuntimeError("cannot finish while the final frame is unacknowledged")
        self.finished = True

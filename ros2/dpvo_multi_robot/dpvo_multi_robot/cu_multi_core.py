"""Pure image and calibration helpers for CU-Multi RGB playback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _field(message: Any, name: str) -> Any:
    """Read a ROS field from ROS1- or ROS2-style rosbags dataclasses."""

    if hasattr(message, name):
        return getattr(message, name)
    upper = name.upper()
    if hasattr(message, upper):
        return getattr(message, upper)
    raise AttributeError(f"message has neither {name!r} nor {upper!r}")


@dataclass(frozen=True)
class PinholeCalibration:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    distortion_model: str


@dataclass(frozen=True)
class PinholeRectification:
    calibration: PinholeCalibration
    camera_matrix: np.ndarray
    map_x: np.ndarray | None
    map_y: np.ndarray | None

    def rectify(self, image: np.ndarray) -> np.ndarray:
        expected = (self.calibration.height, self.calibration.width)
        if image.shape[:2] != expected:
            raise ValueError(f"CU-Multi image has shape {image.shape}, expected {expected}")
        if self.map_x is None or self.map_y is None:
            return np.ascontiguousarray(image)
        return np.ascontiguousarray(
            cv2.remap(
                image,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )


def calibration_from_camera_info(message: Any) -> PinholeCalibration:
    """Copy the calibration fields from a ROS CameraInfo message."""

    width = int(message.width)
    height = int(message.height)
    matrix = np.asarray(_field(message, "k"), dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(_field(message, "d"), dtype=np.float64).reshape(-1)
    model = str(message.distortion_model).lower()
    if width <= 0 or height <= 0:
        raise ValueError("CameraInfo width and height must be positive")
    if not np.isfinite(matrix).all() or matrix[2, 2] == 0.0:
        raise ValueError("CameraInfo contains an invalid camera matrix")
    if not np.isfinite(distortion).all():
        raise ValueError("CameraInfo contains non-finite distortion")
    if model not in ("", "plumb_bob", "rational_polynomial"):
        raise ValueError(f"unsupported CU-Multi distortion model {model!r}")
    if distortion.size not in (0, 4, 5, 8, 12, 14):
        raise ValueError(
            f"unsupported CU-Multi distortion coefficient count {distortion.size}"
        )
    return PinholeCalibration(width, height, matrix, distortion, model or "plumb_bob")


def build_pinhole_rectification(
    calibration: PinholeCalibration, alpha: float = 0.0
) -> PinholeRectification:
    """Create full-resolution OpenCV pinhole/radtan rectification maps."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("rectification alpha must be in [0, 1]")
    distortion = calibration.distortion
    if distortion.size == 0 or np.allclose(distortion, 0.0):
        return PinholeRectification(
            calibration=calibration,
            camera_matrix=calibration.camera_matrix.copy(),
            map_x=None,
            map_y=None,
        )
    size = (calibration.width, calibration.height)
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        distortion,
        size,
        float(alpha),
        size,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        calibration.camera_matrix,
        distortion,
        np.eye(3, dtype=np.float64),
        new_matrix,
        size,
        cv2.CV_32FC1,
    )
    return PinholeRectification(calibration, new_matrix, map_x, map_y)


def decode_raw_image(message: Any) -> tuple[np.ndarray, str]:
    """Decode a ROS Image into contiguous BGR8 or mono8 pixels."""

    width = int(message.width)
    height = int(message.height)
    encoding = str(message.encoding).lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "8uc3": 3,
        "mono8": 1,
        "8uc1": 1,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported CU-Multi image encoding {message.encoding!r}")
    channels = channels_by_encoding[encoding]
    row_bytes = width * channels
    step = int(message.step)
    if width <= 0 or height <= 0 or step < row_bytes:
        raise ValueError(
            f"invalid raw image dimensions width={width}, height={height}, step={step}"
        )
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    required = step * height
    if raw.size < required:
        raise ValueError(f"raw image has {raw.size} bytes, expected at least {required}")
    pixels = raw[:required].reshape(height, step)[:, :row_bytes]
    if channels == 1:
        return np.ascontiguousarray(pixels.reshape(height, width)), "mono8"
    pixels = pixels.reshape(height, width, channels)
    if encoding == "rgb8":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(pixels), "bgr8"


def rectified_camera_info(rectification: PinholeRectification) -> dict[str, Any]:
    """Return distortion-free CameraInfo fields for a rectified image."""

    matrix = rectification.camera_matrix
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    return {
        "width": rectification.calibration.width,
        "height": rectification.calibration.height,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "d": [],
        "distortion_model": "plumb_bob",
    }

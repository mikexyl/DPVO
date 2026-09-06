"""Pure ScaleMaster dataset parsing helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


FRAME_PATTERN = re.compile(r"^frame_(\d+)$")


@dataclass(frozen=True)
class ScaleMasterPose:
    frame_id: int
    timestamp: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class ScaleMasterFrame:
    frame_id: int
    timestamp: float
    image_path: Path


def read_camera_matrix(path: Path) -> np.ndarray:
    """Read and validate a ScaleMaster ``camera_matrix.csv`` file."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ScaleMaster camera matrix does not exist: {path}")
    matrix = np.asarray(np.loadtxt(path, delimiter=","), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"ScaleMaster camera matrix must be finite 3x3: {path}")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError(f"ScaleMaster focal lengths must be positive: {path}")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"ScaleMaster camera matrix has invalid final row: {path}")
    return matrix


def read_odometry(path: Path) -> list[ScaleMasterPose]:
    """Read raw or optimized ScaleMaster ARKit camera poses."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ScaleMaster odometry does not exist: {path}")
    required = ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")
    rows = []
    seen = set()
    previous_timestamp = None
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"ScaleMaster odometry has an invalid header: {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_id = int(row["frame"])
                timestamp = float(row["timestamp"])
                translation = np.asarray(
                    [row["x"], row["y"], row["z"]], dtype=np.float64
                )
                quaternion = np.asarray(
                    [row["qx"], row["qy"], row["qz"], row["qw"]],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid ScaleMaster odometry at {path}:{line_number}"
                ) from error
            norm = float(np.linalg.norm(quaternion))
            if (
                frame_id < 0
                or frame_id in seen
                or not np.isfinite(timestamp)
                or not np.isfinite(translation).all()
                or not np.isfinite(norm)
                or norm < 1e-12
            ):
                raise ValueError(
                    f"invalid ScaleMaster odometry at {path}:{line_number}"
                )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(
                    f"non-increasing ScaleMaster timestamp at {path}:{line_number}"
                )
            seen.add(frame_id)
            previous_timestamp = timestamp
            rows.append(
                ScaleMasterPose(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    translation=translation,
                    quaternion_xyzw=quaternion / norm,
                )
            )
    if not rows:
        raise RuntimeError(f"no ScaleMaster odometry rows in {path}")
    return rows


def read_frames(
    sequence_dir: Path, odometry_name: str = "odometry.csv"
) -> list[ScaleMasterFrame]:
    """Join RGB images to capture timestamps by their explicit frame IDs."""

    sequence_dir = Path(sequence_dir).expanduser()
    frames_dir = sequence_dir / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"ScaleMaster frames directory does not exist: {frames_dir}")
    poses = {row.frame_id: row for row in read_odometry(sequence_dir / odometry_name)}
    images = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(frames_dir.glob(extension))
    indexed = []
    seen = set()
    for image_path in images:
        match = FRAME_PATTERN.match(image_path.stem)
        if match is None:
            raise ValueError(f"unexpected ScaleMaster frame name: {image_path.name}")
        frame_id = int(match.group(1))
        if frame_id in seen:
            raise ValueError(f"duplicate ScaleMaster image frame {frame_id}")
        if frame_id not in poses:
            raise ValueError(
                f"ScaleMaster image frame {frame_id} has no row in {odometry_name}"
            )
        seen.add(frame_id)
        indexed.append((frame_id, image_path))
    if not indexed:
        raise RuntimeError(f"no ScaleMaster RGB images in {frames_dir}")
    indexed.sort(key=lambda item: item[0])
    frames = [
        ScaleMasterFrame(frame_id, poses[frame_id].timestamp, image_path)
        for frame_id, image_path in indexed
    ]
    if any(right.timestamp <= left.timestamp for left, right in zip(frames, frames[1:])):
        raise ValueError("ScaleMaster image timestamps are not strictly increasing")
    return frames

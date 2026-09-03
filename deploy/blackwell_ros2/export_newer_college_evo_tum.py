#!/usr/bin/env python3
"""Export Newer College Base ground truth and staged DPVO solutions for evo."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ros2.dpvo_multi_robot.dpvo_multi_robot.newer_college_core import (
    read_kalibr_camera_calibration,
)


# HALO URDF: Base coincides with alphasense_mount. imu_link is a fixed child at
# xyz=(0.006, -0.00855, 0.0174), rpy=(pi, 0, 0). T_BASE_AS_IMU maps AS-IMU
# coordinates into Base coordinates (the pose of AS-IMU expressed in Base).
T_BASE_AS_IMU = np.eye(4, dtype=np.float64)
T_BASE_AS_IMU[:3, :3] = Rotation.from_euler("xyz", [np.pi, 0.0, 0.0]).as_matrix()
T_BASE_AS_IMU[:3, 3] = [0.006, -0.00855, 0.0174]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--groundtruth",
        action="append",
        metavar="ROBOT=CSV",
        required=True,
        help="Repeat once per robot in the requested scenario.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled_paths(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--groundtruth expects ROBOT=CSV, got {value!r}")
        robot, path_value = value.split("=", 1)
        if robot in paths:
            raise ValueError(f"duplicate ground truth for {robot}")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[robot] = path
    if len(paths) < 2:
        raise ValueError("at least two ground-truth paths are required")
    return paths


def robot_sort_key(robot: str):
    suffix = robot.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot)


def tum_line(timestamp: float, translation: np.ndarray, quaternion: np.ndarray) -> str:
    return " ".join(
        f"{value:.12g}"
        for value in (timestamp, *translation.tolist(), *quaternion.tolist())
    )


def write_tum(path: Path, rows: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
    path.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        + "\n".join(tum_line(*row) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def numeric_fields(line: str) -> list[float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = [field.strip() for field in stripped.replace(",", " ").split()]
    try:
        return [float(field) for field in fields]
    except ValueError:
        # Official files may include a single CSV header row.
        return None


def read_base_groundtruth(
    path: Path, t_base_cam: np.ndarray
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    rows = []
    previous_timestamp = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = numeric_fields(line)
        if fields is None:
            continue
        if len(fields) < 9:
            raise ValueError(
                f"{path}:{line_number} has {len(fields)} numeric columns, expected 9"
            )
        seconds = int(fields[0])
        nanoseconds = int(fields[1])
        if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
            raise ValueError(f"invalid epoch stamp at {path}:{line_number}")
        timestamp = seconds + nanoseconds * 1e-9
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(f"non-increasing ground truth at {path}:{line_number}")
        previous_timestamp = timestamp
        translation_wb = np.asarray(fields[2:5], dtype=np.float64)
        quaternion_wb = np.asarray(fields[5:9], dtype=np.float64)
        norm = np.linalg.norm(quaternion_wb)
        if not np.isfinite(translation_wb).all() or not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"invalid Base pose at {path}:{line_number}")
        quaternion_wb /= norm
        t_world_base = np.eye(4, dtype=np.float64)
        t_world_base[:3, :3] = Rotation.from_quat(quaternion_wb).as_matrix()
        t_world_base[:3, 3] = translation_wb
        t_world_cam = t_world_base @ t_base_cam
        rows.append(
            (
                timestamp,
                t_world_cam[:3, 3].copy(),
                Rotation.from_matrix(t_world_cam[:3, :3]).as_quat(),
            )
        )
    if not rows:
        raise RuntimeError(f"no ground-truth poses in {path}")
    return rows


def read_graph(path: Path) -> tuple[dict[int, dict], dict[str, list[tuple]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    metadata = {}
    grouped = defaultdict(list)
    for vertex in document["vertices"]:
        vertex_id = int(vertex["id"])
        robot = str(vertex["robot_id"])
        estimate = vertex["estimate"]
        timestamp = float(vertex["timestamp"])
        metadata[vertex_id] = {
            "robot_id": robot,
            "timestamp": timestamp,
            "keyframe_id": int(vertex["keyframe_id"]),
        }
        grouped[robot].append(
            (
                timestamp,
                np.asarray(estimate["translation"], dtype=np.float64),
                np.asarray(estimate["quaternion_xyzw"], dtype=np.float64),
            )
        )
    for values in grouped.values():
        values.sort(key=lambda row: row[0])
    return metadata, dict(grouped)


def read_solver_csv(path: Path, metadata: dict[int, dict]) -> dict[str, list[tuple]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            vertex_id = int(row["vertex_id"])
            if vertex_id not in metadata:
                raise ValueError(f"unknown vertex {vertex_id} in {path}")
            vertex = metadata[vertex_id]
            if row["robot_id"] != vertex["robot_id"]:
                raise ValueError(f"robot mismatch for vertex {vertex_id} in {path}")
            scale = float(row["scale"])
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"invalid solver scale for vertex {vertex_id} in {path}")
            grouped[row["robot_id"]].append(
                (
                    vertex["timestamp"],
                    np.asarray([row["tx"], row["ty"], row["tz"]], dtype=np.float64),
                    np.asarray(
                        [row["qx"], row["qy"], row["qz"], row["qw"]],
                        dtype=np.float64,
                    ),
                )
            )
    for values in grouped.values():
        values.sort(key=lambda row: row[0])
    return dict(grouped)


def write_grouped(
    output_dir: Path,
    name: str,
    grouped: dict[str, list[tuple]],
    robots: tuple[str, ...],
) -> None:
    joint = []
    for index, robot in enumerate(robots):
        if robot not in grouped or not grouped[robot]:
            raise RuntimeError(f"{name} has no trajectory for {robot}")
        rows = grouped[robot]
        write_tum(output_dir / f"{name}_{robot}.tum", rows)
        # Keep joint association unambiguous even if collection timestamps
        # overlap. Per-robot files retain the authoritative camera epoch time.
        offset = index * 1_000_000.0
        joint.extend(
            (timestamp + offset, translation, quaternion)
            for timestamp, translation, quaternion in rows
        )
    joint.sort(key=lambda row: row[0])
    write_tum(output_dir / f"{name}_joint.tum", joint)


def main() -> None:
    args = parse_args()
    graph_path = args.graph.expanduser().resolve()
    solver_dir = args.solver_dir.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_paths = parse_labeled_paths(args.groundtruth)
    robots = tuple(sorted(groundtruth_paths, key=robot_sort_key))

    calibration = read_kalibr_camera_calibration(calibration_path, "cam0")
    # Kalibr T_cam_imu maps AS-IMU coordinates into cam0. The camera pose in
    # Base is therefore T_BASE_AS_IMU * inverse(T_CAM0_AS_IMU).
    t_base_cam = T_BASE_AS_IMU @ np.linalg.inv(calibration.t_cam_imu)
    groundtruth = {
        robot: read_base_groundtruth(path, t_base_cam)
        for robot, path in groundtruth_paths.items()
    }
    write_grouped(output_dir, "groundtruth", groundtruth, robots)

    metadata, raw = read_graph(graph_path)
    if set(raw) != set(robots):
        raise RuntimeError(
            f"graph robots {sorted(raw)} do not match ground truth {list(robots)}"
        )
    solutions = {
        "raw": raw,
        "centralized": read_solver_csv(solver_dir / "centralized.csv", metadata),
        "centralized_explicit_anchors": read_solver_csv(
            solver_dir / "centralized_explicit_anchors.csv", metadata
        ),
        "cbs": read_solver_csv(
            solver_dir / "cbs.csv", metadata
        ),
    }
    for name, grouped in solutions.items():
        write_grouped(output_dir, name, grouped, robots)

    manifest = {
        "format": "dpvo_newer_college_evo_export",
        "version": 1,
        "robots": list(robots),
        "groundtruth_input_frame": "Newer College multi-camera Base",
        "groundtruth_output_frame": "AS C0 frontal-right optical frame",
        "transform_convention": (
            "T_WORLD_CAM0 = T_WORLD_BASE * T_BASE_AS_IMU * inverse(T_CAM0_AS_IMU)"
        ),
        "t_base_as_imu": T_BASE_AS_IMU.tolist(),
        "t_cam0_as_imu": calibration.t_cam_imu.tolist(),
        "t_base_cam0": t_base_cam.tolist(),
        "calibration": {
            "path": str(calibration_path),
            "sha256": sha256(calibration_path),
        },
        "graph": {"path": str(graph_path), "sha256": sha256(graph_path)},
        "groundtruth": {
            robot: {
                "path": str(path),
                "sha256": sha256(path),
                "pose_count": len(groundtruth[robot]),
            }
            for robot, path in groundtruth_paths.items()
        },
        "estimate_pose_counts": {
            name: {robot: len(grouped[robot]) for robot in robots}
            for name, grouped in solutions.items()
        },
        "solutions": list(solutions),
        "joint_timestamp_offsets_seconds": {
            robot: index * 1_000_000.0 for index, robot in enumerate(robots)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

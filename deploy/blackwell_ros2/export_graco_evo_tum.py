#!/usr/bin/env python3
"""Export GrAco IMU ground truth and staged cam0 DPVO solutions for evo."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def _load_common_module():
    path = Path(__file__).with_name("export_newer_college_evo_tum.py")
    spec = importlib.util.spec_from_file_location("_dpvo_evo_export_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evo export helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_common_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--groundtruth",
        action="append",
        metavar="ROBOT=TUM",
        required=True,
        help="Map each DPVO robot id to its GrAco IMU-frame reference file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_t_imu_cam0(path: Path) -> np.ndarray:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    record = document.get("T_Imu_cam0") if isinstance(document, dict) else None
    if not isinstance(record, dict) or record.get("rows") != 4 or record.get("cols") != 4:
        raise ValueError(f"{path} does not contain a 4x4 T_Imu_cam0")
    matrix = np.asarray(record.get("data"), dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"invalid homogeneous T_Imu_cam0 in {path}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"T_Imu_cam0 rotation is not orthonormal in {path}")
    return matrix


def read_imu_groundtruth(
    path: Path, t_imu_cam0: np.ndarray
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    rows = []
    previous_timestamp = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = COMMON.numeric_fields(line)
        if fields is None:
            continue
        if len(fields) < 8:
            raise ValueError(
                f"{path}:{line_number} has {len(fields)} numeric columns, expected 8"
            )
        timestamp = float(fields[0])
        if not np.isfinite(timestamp) or (
            previous_timestamp is not None and timestamp <= previous_timestamp
        ):
            raise ValueError(f"invalid or non-increasing timestamp at {path}:{line_number}")
        previous_timestamp = timestamp
        translation = np.asarray(fields[1:4], dtype=np.float64)
        quaternion = np.asarray(fields[4:8], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(translation).all() or not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"invalid reference pose at {path}:{line_number}")
        t_base_imu = np.eye(4, dtype=np.float64)
        t_base_imu[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
        t_base_imu[:3, 3] = translation
        t_base_cam0 = t_base_imu @ t_imu_cam0
        rows.append(
            (
                timestamp,
                t_base_cam0[:3, 3].copy(),
                Rotation.from_matrix(t_base_cam0[:3, :3]).as_quat(),
            )
        )
    if not rows:
        raise RuntimeError(f"no GrAco ground-truth poses in {path}")
    return rows


def main() -> None:
    args = parse_args()
    graph_path = args.graph.expanduser().resolve()
    solver_dir = args.solver_dir.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_paths = COMMON.parse_labeled_paths(args.groundtruth)
    robots = tuple(sorted(groundtruth_paths, key=COMMON.robot_sort_key))

    t_imu_cam0 = read_t_imu_cam0(calibration_path)
    groundtruth = {
        robot: read_imu_groundtruth(path, t_imu_cam0)
        for robot, path in groundtruth_paths.items()
    }
    COMMON.write_grouped(output_dir, "groundtruth", groundtruth, robots)

    metadata, raw = COMMON.read_graph(graph_path)
    if set(raw) != set(robots):
        raise RuntimeError(
            f"graph robots {sorted(raw)} do not match ground truth {list(robots)}"
        )
    cbs_solution = "cbs"
    solutions = {
        "raw": raw,
        "centralized": COMMON.read_solver_csv(
            solver_dir / "centralized.csv", metadata
        ),
        "centralized_explicit_anchors": COMMON.read_solver_csv(
            solver_dir / "centralized_explicit_anchors.csv", metadata
        ),
        cbs_solution: COMMON.read_solver_csv(
            solver_dir / f"{cbs_solution}.csv", metadata
        ),
    }
    for name, grouped in solutions.items():
        COMMON.write_grouped(output_dir, name, grouped, robots)

    manifest = {
        "format": "dpvo_graco_evo_export",
        "version": 1,
        "robots": list(robots),
        "groundtruth_input_frame": "GrAco RTK Base ENU to vehicle IMU",
        "groundtruth_output_frame": "GrAco left camera optical frame (cam0)",
        "transform_convention": "T_BASE_CAM0 = T_BASE_IMU * T_IMU_CAM0",
        "t_imu_cam0": t_imu_cam0.tolist(),
        "calibration": {
            "path": str(calibration_path),
            "sha256": COMMON.sha256(calibration_path),
        },
        "graph": {"path": str(graph_path), "sha256": COMMON.sha256(graph_path)},
        "groundtruth": {
            robot: {
                "path": str(path),
                "sha256": COMMON.sha256(path),
                "pose_count": len(groundtruth[robot]),
            }
            for robot, path in groundtruth_paths.items()
        },
        "estimate_pose_counts": {
            name: {robot: len(grouped[robot]) for robot in robots}
            for name, grouped in solutions.items()
        },
        "solutions": list(solutions),
        "cbs_reference_frame": "robot0_from_each_robot_relative_anchor_estimate",
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

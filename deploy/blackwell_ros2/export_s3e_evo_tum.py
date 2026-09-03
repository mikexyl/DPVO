#!/usr/bin/env python3
"""Export S3E position ground truth and staged DPVO solutions for evo."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


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
    parser.add_argument(
        "--groundtruth", action="append", metavar="ROBOT=TUM", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_position_groundtruth(
    path: Path,
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
        translation = np.asarray(fields[1:4], dtype=np.float64)
        quaternion = np.asarray(fields[4:8], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if (
            not np.isfinite(timestamp)
            or not np.isfinite(translation).all()
            or not np.isfinite(norm)
            or norm < 1e-12
        ):
            raise ValueError(f"invalid ground truth at {path}:{line_number}")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(f"non-increasing timestamp at {path}:{line_number}")
        previous_timestamp = timestamp
        rows.append((timestamp, translation, quaternion / norm))
    if not rows:
        raise RuntimeError(f"no S3E ground-truth poses in {path}")
    return rows


def main() -> None:
    args = parse_args()
    graph_path = args.graph.expanduser().resolve()
    solver_dir = args.solver_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_paths = COMMON.parse_labeled_paths(args.groundtruth)
    robots = tuple(sorted(groundtruth_paths, key=COMMON.robot_sort_key))

    absolute_groundtruth = {
        robot: read_position_groundtruth(path)
        for robot, path in groundtruth_paths.items()
    }
    origin = absolute_groundtruth[robots[0]][0][1].copy()
    groundtruth = {
        robot: [
            (timestamp, translation - origin, quaternion)
            for timestamp, translation, quaternion in rows
        ]
        for robot, rows in absolute_groundtruth.items()
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

    identity_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0])
    orientations_are_identity = all(
        np.allclose(quaternion, identity_quaternion)
        for rows in absolute_groundtruth.values()
        for _timestamp, _translation, quaternion in rows
    )
    manifest = {
        "format": "dpvo_s3e_evo_export",
        "version": 1,
        "robots": list(robots),
        "groundtruth_input_frame": "S3E shared projected position frame",
        "groundtruth_output_frame": "common-origin position frame",
        "common_origin": origin.tolist(),
        "position_only_evaluation": True,
        "groundtruth_orientations_all_identity": orientations_are_identity,
        "camera_lever_arm_applied": False,
        "camera_lever_arm_note": (
            "Ground-truth files contain no platform attitude, so camera-frame "
            "lever-arm rotation cannot be applied reliably."
        ),
        "graph": {"path": str(graph_path), "sha256": COMMON.sha256(graph_path)},
        "groundtruth": {
            robot: {
                "path": str(path),
                "sha256": COMMON.sha256(path),
                "pose_count": len(absolute_groundtruth[robot]),
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

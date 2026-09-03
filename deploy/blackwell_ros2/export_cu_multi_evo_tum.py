#!/usr/bin/env python3
"""Export CU-Multi UTM ground truth and staged DPVO solutions for evo."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


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
        "--groundtruth",
        action="append",
        metavar="ROBOT=CSV",
        required=True,
        help="Map each DPVO robot id to its CU-Multi UTM reference CSV.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_utm_groundtruth(path: Path) -> list[tuple[float, np.ndarray, np.ndarray]]:
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
        rows.append((timestamp, translation, quaternion / norm))
    if not rows:
        raise RuntimeError(f"no CU-Multi ground-truth poses in {path}")
    return rows


def translate_rows(rows: list[tuple], origin: np.ndarray) -> list[tuple]:
    return [
        (timestamp, translation - origin, quaternion)
        for timestamp, translation, quaternion in rows
    ]


def main() -> None:
    args = parse_args()
    graph_path = args.graph.expanduser().resolve()
    solver_dir = args.solver_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth_paths = COMMON.parse_labeled_paths(args.groundtruth)
    robots = tuple(sorted(groundtruth_paths, key=COMMON.robot_sort_key))

    absolute_groundtruth = {
        robot: read_utm_groundtruth(path)
        for robot, path in groundtruth_paths.items()
    }
    # A single common origin retains the globally aligned multi-robot geometry
    # while avoiding precision loss from UTM eastings/northings near 1e6 m.
    origin = absolute_groundtruth[robots[0]][0][1].copy()
    groundtruth = {
        robot: translate_rows(rows, origin)
        for robot, rows in absolute_groundtruth.items()
    }
    COMMON.write_grouped(output_dir, "groundtruth", groundtruth, robots)

    metadata, raw = COMMON.read_graph(graph_path)
    if set(raw) != set(robots):
        raise RuntimeError(
            f"graph robots {sorted(raw)} do not match ground truth {list(robots)}"
        )
    solutions = {
        "raw": raw,
        "centralized": COMMON.read_solver_csv(
            solver_dir / "centralized.csv", metadata
        ),
        "centralized_explicit_anchors": COMMON.read_solver_csv(
            solver_dir / "centralized_explicit_anchors.csv", metadata
        ),
        "cbs": COMMON.read_solver_csv(
            solver_dir / "cbs.csv", metadata
        ),
    }
    for name, grouped in solutions.items():
        COMMON.write_grouped(output_dir, name, grouped, robots)

    manifest = {
        "format": "dpvo_cu_multi_evo_export",
        "version": 1,
        "robots": list(robots),
        "groundtruth_input_frame": "CU-Multi UTM base_link/LiDAR reference",
        "groundtruth_output_frame": "common-origin translated UTM base_link proxy",
        "camera_extrinsic": {
            "status": "unavailable_in_local_dataset_copy",
            "applied": False,
            "evaluation_note": (
                "Translation ATE uses the platform reference pose as a camera-center "
                "proxy; Sim(3) alignment is still applied."
            ),
        },
        "common_utm_origin_xyz_m": origin.tolist(),
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

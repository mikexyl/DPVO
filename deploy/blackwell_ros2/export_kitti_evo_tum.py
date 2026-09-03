#!/usr/bin/env python3
"""Export multi-robot KITTI ground truth and DPVO/CBS trajectories for evo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_WINDOWS = ((0, 1450), (1450, 3000), (3000, 4541))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--tag")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--graph",
        type=Path,
        help="Unoptimized graph; defaults to the online or staged run layout.",
    )
    parser.add_argument(
        "--solver-dir",
        type=Path,
        help="Directory containing centralized.csv and the single cbs.csv.",
    )
    parser.add_argument(
        "--cbs-solution-dir",
        type=Path,
        help="Read cbs.csv from this directory instead of <result>/<tag>_cbs.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        metavar="FRAME",
        default=[value for window in DEFAULT_WINDOWS for value in window],
        help="START END pairs, one pair per robot in index order.",
    )
    return parser.parse_args()


def tum_line(timestamp: float, translation: np.ndarray, quaternion: np.ndarray) -> str:
    values = (timestamp, *translation.tolist(), *quaternion.tolist())
    return " ".join(f"{value:.12g}" for value in values)


def write_tum(path: Path, rows: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
    path.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        + "\n".join(tum_line(*row) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def read_kitti_groundtruth(
    dataset_root: Path, sequence: str
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    pose_path = dataset_root / "poses" / f"{sequence}.txt"
    times_path = dataset_root / "sequences" / sequence / "times.txt"
    poses = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 3, 4)
    timestamps = np.loadtxt(times_path, dtype=np.float64).reshape(-1)
    if len(poses) != len(timestamps):
        raise RuntimeError(
            f"KITTI pose/timestamp mismatch: {len(poses)} versus {len(timestamps)}"
        )
    return [
        (timestamp, pose[:, 3], Rotation.from_matrix(pose[:, :3]).as_quat())
        for timestamp, pose in zip(timestamps, poses, strict=True)
    ]


def read_graph(path: Path) -> tuple[dict[int, dict], dict[str, list[tuple]]]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    metadata = {}
    grouped = defaultdict(list)
    for vertex in graph["vertices"]:
        vertex_id = int(vertex["id"])
        robot = vertex["robot_id"]
        estimate = vertex["estimate"]
        metadata[vertex_id] = {
            "robot_id": robot,
            "timestamp": float(vertex["timestamp"]),
            "keyframe_id": int(vertex["keyframe_id"]),
        }
        grouped[robot].append(
            (
                float(vertex["timestamp"]),
                np.asarray(estimate["translation"], dtype=np.float64),
                np.asarray(estimate["quaternion_xyzw"], dtype=np.float64),
            )
        )
    for rows in grouped.values():
        rows.sort(key=lambda row: row[0])
    return metadata, dict(grouped)


def read_solver_csv(
    path: Path, vertex_metadata: dict[int, dict]
) -> dict[str, list[tuple]]:
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            vertex_id = int(row["vertex_id"])
            metadata = vertex_metadata[vertex_id]
            robot = row["robot_id"]
            if robot != metadata["robot_id"]:
                raise RuntimeError(f"robot mismatch for vertex {vertex_id} in {path}")
            grouped[robot].append(
                (
                    metadata["timestamp"],
                    np.asarray([row["tx"], row["ty"], row["tz"]], dtype=np.float64),
                    np.asarray(
                        [row["qx"], row["qy"], row["qz"], row["qw"]],
                        dtype=np.float64,
                    ),
                )
            )
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
    return dict(grouped)


def write_grouped(
    output_dir: Path,
    name: str,
    grouped: dict[str, list[tuple]],
    robots: tuple[str, ...],
) -> None:
    joint = []
    for robot_index, robot in enumerate(robots):
        rows = grouped[robot]
        write_tum(output_dir / f"{name}_{robot}.tum", rows)
        # Overlapping robot partitions reuse the original KITTI timestamps.
        # Offset timestamps only in the joint file so evo cannot associate a
        # pose from one robot with ground truth from another robot. Per-robot
        # files retain the dataset timestamps unchanged.
        timestamp_offset = robot_index * 1_000_000.0
        joint.extend(
            (timestamp + timestamp_offset, translation, quaternion)
            for timestamp, translation, quaternion in rows
        )
    joint.sort(key=lambda row: row[0])
    write_tum(output_dir / f"{name}_joint.tum", joint)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    tag = args.tag or result_dir.name
    output_dir = (args.output_dir or result_dir / "evo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.windows) % 2:
        raise ValueError("--windows requires START END pairs")
    windows = list(zip(args.windows[::2], args.windows[1::2], strict=True))
    robots = tuple(f"robot{index}" for index in range(len(windows)))
    if not robots:
        raise ValueError("--windows requires at least one robot window")
    all_groundtruth = read_kitti_groundtruth(
        args.dataset_root.resolve(), args.sequence
    )
    groundtruth = {
        robot: all_groundtruth[start:end]
        for robot, (start, end) in zip(robots, windows, strict=True)
    }
    write_grouped(output_dir, "groundtruth", groundtruth, robots)

    unoptimized_path = (
        args.graph.resolve()
        if args.graph
        else result_dir / f"{tag}_pose_graph_keyframes_unoptimized.json"
    )
    if not unoptimized_path.is_file():
        staged_graph = (
            result_dir
            / "geometric_verification"
            / "unoptimized_verified_graph.json"
        )
        if staged_graph.is_file():
            unoptimized_path = staged_graph
    optimized_path = result_dir / f"{tag}_pose_graph_keyframes.json"
    cbs_dir = (
        args.solver_dir.resolve()
        if args.solver_dir
        else result_dir / f"{tag}_cbs"
    )
    if not cbs_dir.is_dir() and (result_dir / "dpgo").is_dir():
        cbs_dir = result_dir / "dpgo"
    cbs_solution_dir = (
        args.cbs_solution_dir.resolve() if args.cbs_solution_dir else cbs_dir
    )
    cbs_input = cbs_dir / "input_keyframes_unoptimized.json"

    vertex_metadata, raw = read_graph(unoptimized_path)
    solutions = {
        "raw": raw,
        "full_graph_centralized": read_solver_csv(
            cbs_dir / "centralized.csv", vertex_metadata
        ),
        "cbs": read_solver_csv(
            cbs_solution_dir / "cbs.csv", vertex_metadata
        ),
    }
    if optimized_path.is_file():
        _, online_centralized = read_graph(optimized_path)
        solutions = {
            "raw": solutions["raw"],
            "online_centralized": online_centralized,
            "full_graph_centralized": solutions["full_graph_centralized"],
            "cbs": solutions["cbs"],
        }
    for name, grouped in solutions.items():
        missing = set(robots) - set(grouped)
        if missing:
            raise RuntimeError(f"{name} is missing trajectories for {sorted(missing)}")
        write_grouped(output_dir, name, grouped, robots)

    raw_hash = sha256(unoptimized_path)
    cbs_hash = sha256(cbs_input)
    manifest = {
        "run": tag,
        "dataset_root": str(args.dataset_root.resolve()),
        "sequence": args.sequence,
        "robots": list(robots),
        "windows": {robot: list(window) for robot, window in zip(robots, windows)},
        "groundtruth_usage": "split selection and post-run evaluation only",
        "groundtruth_pose_counts": {
            robot: len(groundtruth[robot]) for robot in robots
        },
        "estimate_pose_counts": {
            name: {robot: len(grouped[robot]) for robot in robots}
            for name, grouped in solutions.items()
        },
        "raw_graph_sha256": raw_hash,
        "cbs_input_graph_sha256": cbs_hash,
        "cbs_input_byte_identical_to_raw_graph": (
            raw_hash == cbs_hash if raw_hash is not None and cbs_hash is not None else None
        ),
        "solutions": list(solutions),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

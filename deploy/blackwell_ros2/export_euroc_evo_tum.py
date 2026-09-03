#!/usr/bin/env python3
"""Export EuRoC ground truth and DPVO/CBS trajectories for evo.

The exported ground truth is the cam0 pose in the dataset world frame. EuRoC's
Vicon topic and ASL ``state_groundtruth_estimate0`` CSV both report the body
pose, so the fixed cam0 ``T_BS`` extrinsic from ``mav0/cam0/sensor.yaml`` is
applied before writing TUM trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation


ROBOTS = ("robot0", "robot1", "robot2")
DEFAULT_BAGS = ("V1_01_easy.bag", "V1_02_medium.bag", "V1_03_difficult.bag")

# EuRoC cam0 sensor extrinsics wrt. the body frame (T_BS).  This exact matrix
# is distributed in mav0/cam0/sensor.yaml for the Vicon Room 1 sequences.
T_BS = np.asarray(
    [
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--bag-root", type=Path, default=Path("/data/euroc"))
    parser.add_argument("--bag0", default=DEFAULT_BAGS[0])
    parser.add_argument("--bag1", default=DEFAULT_BAGS[1])
    parser.add_argument("--bag2", default=DEFAULT_BAGS[2])
    parser.add_argument("--tag")
    parser.add_argument("--output-dir", type=Path)
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


def read_vicon_camera_poses(bag: Path) -> list[tuple[float, np.ndarray, np.ndarray]]:
    with AnyReader([bag]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic.startswith("/vicon/")
            and connection.msgtype == "geometry_msgs/msg/TransformStamped"
        ]
        if len(connections) != 1:
            raise RuntimeError(f"expected one Vicon TransformStamped topic in {bag}")

        rows = []
        for connection, bag_timestamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            stamp = message.header.stamp
            timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            if timestamp == 0.0:
                timestamp = float(bag_timestamp) * 1e-9
            translation_wb = np.asarray(
                [
                    message.transform.translation.x,
                    message.transform.translation.y,
                    message.transform.translation.z,
                ],
                dtype=np.float64,
            )
            quaternion_wb = np.asarray(
                [
                    message.transform.rotation.x,
                    message.transform.rotation.y,
                    message.transform.rotation.z,
                    message.transform.rotation.w,
                ],
                dtype=np.float64,
            )
            rotation_wb = Rotation.from_quat(quaternion_wb).as_matrix()
            rotation_ws = rotation_wb @ T_BS[:3, :3]
            translation_ws = translation_wb + rotation_wb @ T_BS[:3, 3]
            rows.append(
                (
                    timestamp,
                    translation_ws,
                    Rotation.from_matrix(rotation_ws).as_quat(),
                )
            )
    return rows


def read_asl_camera_poses(
    sequence_dir: Path,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    groundtruth_path = (
        sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    )
    if not groundtruth_path.is_file():
        raise FileNotFoundError(groundtruth_path)

    rows = []
    with groundtruth_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(line for line in handle if not line.startswith("#"))
        for fields in reader:
            timestamp = float(fields[0]) * 1e-9
            translation_wb = np.asarray(fields[1:4], dtype=np.float64)
            quaternion_wxyz = np.asarray(fields[4:8], dtype=np.float64)
            quaternion_wb = quaternion_wxyz[[1, 2, 3, 0]]
            rotation_wb = Rotation.from_quat(quaternion_wb).as_matrix()
            rotation_ws = rotation_wb @ T_BS[:3, :3]
            translation_ws = translation_wb + rotation_wb @ T_BS[:3, 3]
            rows.append(
                (
                    timestamp,
                    translation_ws,
                    Rotation.from_matrix(rotation_ws).as_quat(),
                )
            )
    return rows


def read_groundtruth_camera_poses(
    bag: Path,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    asl_sequence = bag.with_suffix("")
    asl_groundtruth = (
        asl_sequence / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    )
    if asl_groundtruth.is_file():
        return read_asl_camera_poses(asl_sequence)
    return read_vicon_camera_poses(bag)


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


def read_solver_csv(path: Path, vertex_metadata: dict[int, dict]) -> dict[str, list[tuple]]:
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
                    np.asarray([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64),
                )
            )
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
    return dict(grouped)


def write_grouped(output_dir: Path, name: str, grouped: dict[str, list[tuple]]) -> None:
    joint = []
    for robot in ROBOTS:
        rows = grouped[robot]
        write_tum(output_dir / f"{name}_{robot}.tum", rows)
        joint.extend(rows)
    joint.sort(key=lambda row: row[0])
    write_tum(output_dir / f"{name}_joint.tum", joint)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    tag = args.tag or result_dir.name
    output_dir = (args.output_dir or result_dir / "evo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bag_paths = [args.bag_root / name for name in (args.bag0, args.bag1, args.bag2)]
    groundtruth = {
        robot: read_groundtruth_camera_poses(path)
        for robot, path in zip(ROBOTS, bag_paths, strict=True)
    }
    write_grouped(output_dir, "groundtruth", groundtruth)

    unoptimized_path = result_dir / f"{tag}_pose_graph_keyframes_unoptimized.json"
    optimized_path = result_dir / f"{tag}_pose_graph_keyframes.json"
    cbs_dir = result_dir / f"{tag}_cbs"

    vertex_metadata, raw = read_graph(unoptimized_path)
    _, online_centralized = read_graph(optimized_path)
    solutions = {
        "raw": raw,
        "online_centralized": online_centralized,
        "full_graph_centralized": read_solver_csv(
            cbs_dir / "centralized.csv", vertex_metadata
        ),
        "cbs": read_solver_csv(
            cbs_dir / "cbs.csv", vertex_metadata
        ),
    }
    for name, grouped in solutions.items():
        write_grouped(output_dir, name, grouped)

    manifest = {
        "run": tag,
        "groundtruth_frame": "EuRoC dataset world to cam0 using T_WS = T_WB * T_BS",
        "bag_paths": [str(path) for path in bag_paths],
        "groundtruth_pose_counts": {
            robot: len(groundtruth[robot]) for robot in ROBOTS
        },
        "estimate_pose_counts": {
            name: {robot: len(grouped[robot]) for robot in ROBOTS}
            for name, grouped in solutions.items()
        },
        "solutions": list(solutions),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

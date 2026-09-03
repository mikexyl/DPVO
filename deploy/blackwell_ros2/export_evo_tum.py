#!/usr/bin/env python3
"""Export timestamped DPVO/CBS trajectories for EVO TUM-format evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--groundtruth0", type=Path)
    parser.add_argument("--groundtruth1", type=Path)
    return parser.parse_args()


def read_ground_truth(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fields = line.split()
            if len(fields) >= 8:
                rows.append((float(fields[0]), *map(float, fields[1:8])))
    return rows


def read_estimate(path: Path, timestamp_by_vertex):
    grouped = {"robot0": [], "robot1": []}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            vertex_id = int(raw["vertex_id"])
            grouped[raw["robot_id"]].append(
                (
                    timestamp_by_vertex[vertex_id],
                    *(
                        float(raw[name])
                        for name in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
                    ),
                )
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: row[0])
    return grouped


def write_tum(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for row in sorted(rows, key=lambda item: item[0]):
            handle.write(" ".join(f"{value:.17g}" for value in row) + "\n")


def main():
    args = parse_args()
    tag = args.result_dir.name
    output_dir = args.output_dir or args.result_dir / "evo"
    graph_path = args.result_dir / f"{tag}_pose_graph_keyframes_unoptimized.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    timestamp_by_vertex = {
        int(vertex["id"]): float(vertex["timestamp"])
        for vertex in graph["vertices"]
    }
    cbs_dir = args.result_dir / f"{tag}_cbs"
    estimates = {
        "centralized": read_estimate(cbs_dir / "centralized.csv", timestamp_by_vertex),
        "cbs": read_estimate(
            cbs_dir / "cbs.csv", timestamp_by_vertex
        ),
    }
    groundtruth_paths = {
        "robot0": args.groundtruth0
        or args.result_dir / "groundtruth" / "fr1_desk.txt",
        "robot1": args.groundtruth1
        or args.result_dir / "groundtruth" / "fr1_desk2.txt",
    }
    ground_truth = {
        robot: read_ground_truth(path) for robot, path in groundtruth_paths.items()
    }

    for robot, rows in ground_truth.items():
        write_tum(output_dir / f"groundtruth_{robot}.tum", rows)
    write_tum(
        output_dir / "groundtruth_joint.tum",
        ground_truth["robot0"] + ground_truth["robot1"],
    )
    for solution, grouped in estimates.items():
        for robot, rows in grouped.items():
            write_tum(output_dir / f"{solution}_{robot}.tum", rows)
        write_tum(
            output_dir / f"{solution}_joint.tum",
            grouped["robot0"] + grouped["robot1"],
        )

    manifest = {
        "source_run": tag,
        "graph": str(graph_path),
        "solutions": {
            solution: {robot: len(rows) for robot, rows in grouped.items()}
            for solution, grouped in estimates.items()
        },
        "ground_truth_rows": {
            robot: len(rows) for robot, rows in ground_truth.items()
        },
        "ground_truth_paths": {
            robot: str(path) for robot, path in groundtruth_paths.items()
        },
        "ground_truth_timestamp_ranges": {
            robot: [rows[0][0], rows[-1][0]]
            for robot, rows in ground_truth.items()
        },
        "joint_alignment_note": (
            "EVO estimates one Sim(3) over the union of both trajectories; "
            "joint evaluation requires the two ground-truth timestamp ranges to be "
            "non-overlapping"
        ),
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

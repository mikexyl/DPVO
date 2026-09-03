#!/usr/bin/env python3
"""Evaluate timestamped DPVO graph trajectories against TUM ground truth."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence0", default="rgbd_dataset_freiburg1_desk")
    parser.add_argument("--sequence1", default="rgbd_dataset_freiburg1_desk2")
    parser.add_argument("--centralized", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-time-delta", type=float, default=0.02)
    return parser.parse_args()


def read_ground_truth(path: Path):
    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        poses.append((float(fields[0]), np.asarray(fields[1:4], dtype=np.float64)))
    return poses


def associate(vertices, ground_truth, max_time_delta):
    timestamps = [item[0] for item in ground_truth]
    estimated = []
    reference = []
    deltas = []
    for vertex in vertices:
        timestamp = float(vertex["timestamp"])
        index = bisect_left(timestamps, timestamp)
        candidates = ground_truth[max(0, index - 1) : min(len(ground_truth), index + 1)]
        candidate = min(candidates, key=lambda item: abs(item[0] - timestamp))
        delta = abs(candidate[0] - timestamp)
        if delta <= max_time_delta:
            estimated.append(vertex["estimate"]["translation"])
            reference.append(candidate[1])
            deltas.append(delta)
    return (
        np.asarray(estimated, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        np.asarray(deltas, dtype=np.float64),
    )


def umeyama(src, dst):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.T @ src_centered / src.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.square(src_centered).sum() / src.shape[0]
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return rotation, translation, scale


def trajectory_metrics(estimated, reference, deltas):
    rotation, translation, scale = umeyama(estimated, reference)
    aligned = (scale * (rotation @ estimated.T)).T + translation
    errors = np.linalg.norm(aligned - reference, axis=1)
    distances = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    return {
        "associations": int(len(errors)),
        "max_time_delta_seconds": float(deltas.max()),
        "alignment_scale_local_to_meters": scale,
        "ate_rmse_meters": float(np.sqrt(np.mean(np.square(errors)))),
        "ate_median_meters": float(np.median(errors)),
        "ate_max_meters": float(errors.max()),
        "ground_truth_path_length_meters": float(distances.sum()),
    }


def main():
    args = parse_args()
    graph = json.loads(args.graph.read_text())
    sequence_names = {
        "robot0": args.sequence0,
        "robot1": args.sequence1,
    }
    report = {"graph": str(args.graph), "robots": {}}
    for robot, sequence in sequence_names.items():
        vertices = sorted(
            (v for v in graph["vertices"] if v["robot_id"] == robot),
            key=lambda vertex: vertex["keyframe_id"],
        )
        ground_truth = read_ground_truth(
            args.dataset_root / sequence / "groundtruth.txt"
        )
        estimated, reference, deltas = associate(
            vertices, ground_truth, args.max_time_delta
        )
        if len(estimated) < 3:
            raise RuntimeError(f"only {len(estimated)} associations for {robot}")
        report["robots"][robot] = {
            "sequence": sequence,
            **trajectory_metrics(estimated, reference, deltas),
        }

    scale0 = report["robots"]["robot0"]["alignment_scale_local_to_meters"]
    scale1 = report["robots"]["robot1"]["alignment_scale_local_to_meters"]
    expected_anchor_scale = scale1 / scale0
    report["relative_scale"] = {
        "ground_truth_inferred_robot1_to_robot0": expected_anchor_scale,
    }
    if args.centralized:
        centralized = json.loads(args.centralized.read_text())
        robot1_key = next(key for key in centralized["robots"] if key.startswith("robot1:"))
        estimated_anchor_scale = float(centralized["robots"][robot1_key]["scale"])
        report["relative_scale"].update(
            {
                "estimated_robot1_to_robot0": estimated_anchor_scale,
                "absolute_error": abs(estimated_anchor_scale - expected_anchor_scale),
                "relative_error_percent": 100.0
                * abs(estimated_anchor_scale / expected_anchor_scale - 1.0),
            }
        )

    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

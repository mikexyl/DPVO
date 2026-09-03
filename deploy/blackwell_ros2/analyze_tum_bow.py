#!/usr/bin/env python3
"""Measure cross-session DBoW2 overlap at exported DPVO keyframes."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path

import cv2
import dpretrieval
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence0", default="rgbd_dataset_freiburg1_desk")
    parser.add_argument("--sequence1", default="rgbd_dataset_freiburg1_desk2")
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--crop-x", type=int, default=16)
    parser.add_argument("--crop-y", type=int, default=8)
    parser.add_argument("--fx", type=float, default=517.3)
    parser.add_argument("--fy", type=float, default=516.5)
    parser.add_argument("--cx", type=float, default=318.6)
    parser.add_argument("--cy", type=float, default=255.3)
    parser.add_argument("--k1", type=float, default=0.2624)
    parser.add_argument("--k2", type=float, default=-0.9531)
    parser.add_argument("--p1", type=float, default=-0.0054)
    parser.add_argument("--p2", type=float, default=0.0026)
    parser.add_argument("--k3", type=float, default=1.1633)
    return parser.parse_args()


def read_rgb_list(sequence_dir: Path):
    frames = []
    for line in (sequence_dir / "rgb.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative_path = line.split()[:2]
        frames.append((float(timestamp), sequence_dir / relative_path))
    return frames


def nearest_frame(frames, timestamp):
    timestamps = [item[0] for item in frames]
    index = bisect_left(timestamps, timestamp)
    candidates = frames[max(0, index - 1) : min(len(frames), index + 1)]
    return min(candidates, key=lambda item: abs(item[0] - timestamp))


def preprocess(path: Path, camera_matrix, distortion, crop_x, crop_y):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode {path}")
    if np.any(distortion):
        image = cv2.undistort(image, camera_matrix, distortion)
    if crop_y:
        image = image[crop_y:-crop_y]
    if crop_x:
        image = image[:, crop_x:-crop_x]
    return np.ascontiguousarray(image)


def sparse_bow(entries):
    if not entries:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.float32)
    word_ids, values = zip(*entries)
    return np.asarray(word_ids, dtype=np.uint32), np.asarray(values, dtype=np.float32)


def bow_score(first, second):
    _, first_idx, second_idx = np.intersect1d(
        first[0], second[0], assume_unique=True, return_indices=True
    )
    return float(np.minimum(first[1][first_idx], second[1][second_idx]).sum())


def main():
    args = parse_args()
    if args.crop_x < 0 or args.crop_y < 0:
        raise ValueError("crop values must be non-negative")
    camera_matrix = np.array(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array(
        [args.k1, args.k2, args.p1, args.p2, args.k3], dtype=np.float64
    )
    graph = json.loads(args.graph.read_text())
    vertices = {
        robot: sorted(
            (v for v in graph["vertices"] if v["robot_id"] == robot),
            key=lambda v: v["keyframe_id"],
        )
        for robot in ("robot0", "robot1")
    }
    source_frames = {
        "robot0": read_rgb_list(args.dataset_root / args.sequence0),
        "robot1": read_rgb_list(args.dataset_root / args.sequence1),
    }

    retrieval = dpretrieval.DPRetrieval(str(args.vocabulary), 50)
    bows = {"robot0": [], "robot1": []}
    frame_metadata = {"robot0": [], "robot1": []}
    for robot in ("robot0", "robot1"):
        for index, vertex in enumerate(vertices[robot], start=1):
            source_timestamp, path = nearest_frame(
                source_frames[robot], vertex["timestamp"]
            )
            entries = retrieval.insert_image(
                preprocess(
                    path,
                    camera_matrix,
                    distortion,
                    args.crop_x,
                    args.crop_y,
                )
            )
            bows[robot].append(sparse_bow(entries))
            frame_metadata[robot].append(
                {
                    "keyframe_id": vertex["keyframe_id"],
                    "graph_timestamp": vertex["timestamp"],
                    "image_timestamp": source_timestamp,
                    "image": str(path),
                }
            )
            if index % 25 == 0 or index == len(vertices[robot]):
                print(f"encoded {robot}: {index}/{len(vertices[robot])}", flush=True)

    scores = np.empty((len(bows["robot0"]), len(bows["robot1"])), np.float32)
    for i, first in enumerate(bows["robot0"]):
        for j, second in enumerate(bows["robot1"]):
            scores[i, j] = bow_score(first, second)

    flat_order = np.argsort(scores, axis=None)[::-1]
    top_pairs = []
    for flat_index in flat_order[: args.top]:
        i, j = np.unravel_index(flat_index, scores.shape)
        top_pairs.append(
            {
                "score": float(scores[i, j]),
                "robot0": frame_metadata["robot0"][i],
                "robot1": frame_metadata["robot1"][j],
            }
        )

    best_for_robot1 = scores.max(axis=0)
    best_robot0_index = scores.argmax(axis=0)
    thresholds = {}
    for threshold in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        accepted = best_for_robot1 >= threshold
        consecutive_pairs = int(np.count_nonzero(accepted[:-1] & accepted[1:]))
        thresholds[str(threshold)] = {
            "remote_keyframes_above": int(np.count_nonzero(accepted)),
            "consecutive_remote_pairs_above": consecutive_pairs,
            "all_pairs_above": int(np.count_nonzero(scores >= threshold)),
        }

    report = {
        "graph": str(args.graph),
        "keyframes": {robot: len(items) for robot, items in vertices.items()},
        "score_summary": {
            "maximum": float(scores.max()),
            "median_best_for_robot1": float(np.median(best_for_robot1)),
            "p90_best_for_robot1": float(np.percentile(best_for_robot1, 90)),
            "p95_best_for_robot1": float(np.percentile(best_for_robot1, 95)),
        },
        "thresholds": thresholds,
        "best_matches_for_robot1": [
            {
                "score": float(best_for_robot1[j]),
                "robot0": frame_metadata["robot0"][int(best_robot0_index[j])],
                "robot1": frame_metadata["robot1"][j],
            }
            for j in range(len(best_for_robot1))
        ],
        "top_pairs": top_pairs,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(json.dumps({k: report[k] for k in ("keyframes", "score_summary", "thresholds")}, indent=2))
    print("top pairs:")
    for pair in top_pairs:
        print(
            f"  {pair['score']:.6f}: "
            f"robot0/{pair['robot0']['keyframe_id']} <-> "
            f"robot1/{pair['robot1']['keyframe_id']}"
        )


if __name__ == "__main__":
    main()

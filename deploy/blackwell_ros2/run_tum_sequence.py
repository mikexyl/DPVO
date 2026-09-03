#!/usr/bin/env python3
"""Replay one TUM RGB-D sequence and export its final DPVO sparse map."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo import lietorch


def read_rgb_list(sequence_dir: Path) -> list[tuple[float, Path]]:
    frames = []
    for line_number, line in enumerate(
        (sequence_dir / "rgb.txt").read_text().splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"malformed rgb.txt line {line_number}: {line}")
        image_path = sequence_dir / fields[1]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        frames.append((float(fields[0]), image_path))
    if not frames:
        raise RuntimeError(f"no RGB frames in {sequence_dir}")
    return frames


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    frames = read_rgb_list(args.sequence_dir)[:: args.stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    camera_matrix = np.asarray(
        [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [args.k1, args.k2, args.p1, args.p2, args.k3], dtype=np.float64
    )
    intrinsics_np = np.asarray(
        [args.fx, args.fy, args.cx - args.crop_x, args.cy - args.crop_y],
        dtype=np.float32,
    )

    slam = None
    for frame_index, (timestamp, image_path) in enumerate(frames, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {image_path}")
        image = cv2.undistort(image, camera_matrix, distortion)
        if args.crop_y:
            image = image[args.crop_y : -args.crop_y]
        if args.crop_x:
            image = image[:, args.crop_x : -args.crop_x]
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        image = image[: height - height % 16, : width - width % 16]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics_np).cuda()
        if slam is None:
            slam = DPVO(
                cfg,
                str(args.network),
                ht=image_tensor.shape[1],
                wd=image_tensor.shape[2],
            )
        slam(timestamp, image_tensor, intrinsics)
        if frame_index % 100 == 0:
            print(f"Processed {frame_index}/{len(frames)}", flush=True)

    if slam is None:
        raise RuntimeError("sequence produced no frames")

    # Snapshot the current state directly. The online ROS nodes skip final BA,
    # and waiting for an in-flight classic-PGO callback can deadlock during an
    # offline replay even though all map state needed here is already complete.
    slam.traj = {
        int(slam.pg.tstamps_[index]): slam.pg.poses_[index]
        for index in range(slam.n)
    }
    poses = lietorch.stack(
        [slam.get_pose(timestamp) for timestamp in range(slam.counter)], dim=0
    )
    poses = poses.inv().data.detach().cpu().numpy()
    timestamps = np.asarray(slam.tlist, dtype=np.float64)
    points = slam.pg.points_.detach().cpu().numpy()[: slam.m]
    colors = slam.pg.colors_.view(-1, 3).detach().cpu().numpy()[: slam.m]
    keyframe_input_indices = slam.pg.tstamps_[: slam.n].astype(np.int64)
    keyframe_timestamps = np.asarray(
        [slam.tlist[int(index)] for index in keyframe_input_indices],
        dtype=np.float64,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        poses=poses,
        timestamps=timestamps,
        points=points,
        colors=colors,
        session_from_map=slam.pg.session_from_map_,
        keyframe_input_indices=keyframe_input_indices,
        keyframe_timestamps=keyframe_timestamps,
        keyframe_count=np.asarray(slam.n, dtype=np.int64),
        patch_count=np.asarray(slam.m, dtype=np.int64),
        patches_per_keyframe=np.asarray(slam.M, dtype=np.int64),
        sequence=np.asarray(args.sequence_dir.name),
    )
    print(
        f"Wrote {args.output}: {slam.n} keyframes, {slam.m} sparse points",
        flush=True,
    )
    # Retrieval/PGO uses non-daemon worker threads. The snapshot is closed and
    # flushed at this point, so avoid re-entering their shutdown callback.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--network", type=Path, default=Path("dpvo.pth"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--orb-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
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
    parser.add_argument("--max-edge-age", type=int, default=48)
    parser.add_argument("--loop-retr-thresh", type=float, default=0.012)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least one")
    if args.crop_x < 0 or args.crop_y < 0:
        parser.error("crop values must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    cfg.merge_from_file(str(args.config))
    cfg.CLASSIC_LOOP_CLOSURE = True
    cfg.CLASSIC_PGO_USE_THREADS = True
    cfg.LOOP_CLOSURE = True
    cfg.MAX_EDGE_AGE = args.max_edge_age
    cfg.LOOP_RETR_THRESH = args.loop_retr_thresh
    cfg.ORB_VOCAB_PATH = str(args.orb_vocab)
    torch.manual_seed(1234)
    run(args)


if __name__ == "__main__":
    main()

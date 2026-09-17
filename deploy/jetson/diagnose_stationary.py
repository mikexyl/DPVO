"""Controlled move-then-stop replay: detect motion hallucinated on identical images."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.rerun_viewer import _invert_poses
from tensorrt_encoder import install_encoders


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--trt-encoders')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)
    cv2.setNumThreads(1)
    cfg.merge_from_file(args.config)
    image = cv2.imread(args.image)
    intrinsics = torch.tensor(np.loadtxt(args.calib).reshape(-1), device='cuda', dtype=torch.float32)
    slam = DPVO(cfg, '/models/dpvo.pth', ht=image.shape[0], wd=image.shape[1])
    if args.trt_encoders:
        install_encoders(slam.network, args.trt_encoders, '/models/dpvo.pth')
    positions = []
    # A translated planar image, followed by exactly identical images: the
    # stationary interval has zero true motion independent of monocular scale.
    for index in range(240):
        shift = min(index, 39) * 3.0
        moved = cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, 0]]),
                               image.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
        tensor = torch.from_numpy(moved).permute(2, 0, 1).cuda()
        slam(index / 15, tensor, intrinsics)
        positions.append(_invert_poses(slam.pg.poses_[slam.n-1:slam.n])[0, :3])
    positions = np.array(positions)
    motion = np.linalg.norm(positions[59] - positions[0])
    drift = np.linalg.norm(positions[-1] - positions[59])
    report = dict(config=args.config, trt=bool(args.trt_encoders), initialized=slam.is_initialized,
                  translation_before_stop=float(motion), translation_after_stop=float(drift),
                  drift_ratio=float(drift / max(motion, 1e-8)), positions=positions.tolist())
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'positions'}), flush=True)


if __name__ == '__main__':
    main()

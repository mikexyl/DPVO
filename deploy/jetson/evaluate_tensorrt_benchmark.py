"""Evaluate benchmark.py trajectories against EuRoC positions with Sim(3) alignment."""
import argparse
import json
from pathlib import Path

import numpy as np


def aligned_errors(estimate, reference):
    x, y = estimate - estimate.mean(0), reference - reference.mean(0)
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(signs) @ vt
    variance = np.mean(np.sum(x*x, axis=1))
    if variance < 1e-12:
        raise ValueError('Trajectory is stationary or degenerate')
    scale = float((singular * signs).sum() / variance)
    error = np.linalg.norm(scale * x @ rotation.T - y, axis=1)
    return dict(sim3_ate_rmse_m=float(np.sqrt(np.mean(error**2))),
                max_error_m=float(error.max()), alignment_scale=scale)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--groundtruth', required=True)
    parser.add_argument('--results', nargs='+', required=True)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))[::args.stride]
    groundtruth = np.loadtxt(args.groundtruth, delimiter=',')
    origin = groundtruth[0,0]
    gt_times = (groundtruth[:,0] - origin) * 1e-9
    report = {}
    for filename in args.results:
        path = Path(filename)
        poses = np.loadtxt(path.with_suffix('.tum'))
        indexes = poses[:,0].astype(int)
        if not np.all(poses[:,0] == indexes) or min(indexes) < 0 or max(indexes) >= len(files):
            raise ValueError('Expected benchmark.py frame indexes as trajectory timestamps')
        times = (np.array([int(files[i].stem) for i in indexes],dtype=np.float64) - origin)*1e-9
        valid = (times >= gt_times[0]) & (times <= gt_times[-1])
        if valid.sum() < 3:
            raise ValueError('Insufficient overlap with ground truth')
        target = np.column_stack([np.interp(times[valid],gt_times,groundtruth[:,k]) for k in (1,2,3)])
        value = aligned_errors(poses[valid,1:4],target)
        value.update(evaluated_frames=int(valid.sum()), excluded_frames=int((~valid).sum()))
        report[path.stem] = value
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()

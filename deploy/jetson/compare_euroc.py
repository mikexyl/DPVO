"""Evaluate benchmark frame-index trajectories against EuRoC camera positions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sequence', required=True, help='EuRoC directory containing mav0')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('trajectories', nargs='+')
    args = parser.parse_args()
    mav = Path(args.sequence) / 'mav0'
    files = sorted((mav / 'cam0/data').glob('*.png'))[::args.stride]
    timestamps = np.array([int(p.stem) for p in files], dtype=np.float64)
    truth = np.loadtxt(mav / 'state_groundtruth_estimate0/data.csv', delimiter=',')
    sensor = yaml.safe_load((mav / 'cam0/sensor.yaml').read_text())
    body_from_camera = np.array(sensor['T_BS']['data']).reshape(4, 4)
    positions = truth[:, 1:4] + Rotation.from_quat(truth[:, [5, 6, 7, 4]]).apply(
        np.broadcast_to(body_from_camera[:3, 3], truth[:, 1:4].shape))
    report = {}
    for filename in args.trajectories:
        estimated = np.loadtxt(filename, ndmin=2)
        indices = estimated[:, 0].astype(int)
        if not np.allclose(indices, estimated[:, 0]) or (indices < 0).any():
            raise ValueError('Expected frame-index timestamps from benchmark.py')
        times = timestamps[indices]
        valid = (times >= truth[0, 0]) & (times <= truth[-1, 0])
        reference = np.column_stack([np.interp(times[valid], truth[:, 0], positions[:, d])
                                     for d in range(3)])
        predicted = estimated[valid, 1:4]
        x, y = predicted - predicted.mean(0), reference - reference.mean(0)
        u, singular, vt = np.linalg.svd(x.T @ y)
        signs = np.ones(3)
        signs[-1] = np.linalg.det(u @ vt)
        rotation = (u * signs) @ vt
        scale = (singular * signs).sum() / np.square(x).sum()
        aligned = scale * x @ rotation + reference.mean(0)
        errors = np.linalg.norm(aligned - reference, axis=1)
        report[Path(filename).stem] = dict(
            matched_frames=len(errors), sim3_scale=float(scale),
            ate_rmse_m=float(np.sqrt(np.mean(errors**2))),
            ate_max_m=float(errors.max()),
            reference_path_m=float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()))
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

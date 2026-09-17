"""Synchronized, headless DPVO timing on an image sequence."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo import fastba


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--target-fps', type=float, default=20.0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--trt-encoders')
    parser.add_argument('--trt-update', help='Opt-in laptop/target-built dense update engines')
    parser.add_argument('--capture-update-inputs', help='Save a few real update inputs for parity checks')
    parser.add_argument('--remap', action='store_true',
                        help='Cache undistortion maps for lower preprocessing cost')
    parser.add_argument('--prefetch', type=int, default=0,
                        help='Bounded number of images to preprocess on a reader thread')
    parser.add_argument('--opts', nargs='*', default=[])
    args = parser.parse_args()
    if (args.warmup < 0 or args.frames <= args.warmup or args.stride < 1
            or args.scale <= 0 or args.target_fps <= 0 or args.prefetch < 0):
        parser.error('Require frames > warmup >= 0, stride >= 1, scale/fps > 0')
    torch.manual_seed(1234)
    np.random.seed(1234)
    cv2.setNumThreads(1)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    original_ba = fastba.BA
    ba_errors = []

    def checked_ba(*positional, **keywords):
        try:
            return original_ba(*positional, **keywords)
        except Exception as error:
            ba_errors.append(repr(error))
            raise

    fastba.BA = checked_ba
    files = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    files = files[::args.stride][:args.frames]
    if len(files) <= args.warmup:
        raise ValueError('Sequence must be longer than warmup')
    calibration = np.loadtxt(args.calib)
    fx, fy, cx, cy = calibration[:4]
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    slam = None
    initialization_frame = None
    times, total_times = [], []
    events, stage_times = {}, {}
    update_samples = []
    sampled_frames = set()
    undistortion_maps = {}
    def read_frame(path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        if len(calibration) > 4:
            if args.remap:
                shape = image.shape[:2]
                if shape not in undistortion_maps:
                    undistortion_maps[shape] = cv2.initUndistortRectifyMap(
                        camera, calibration[4:], None, camera, shape[::-1], cv2.CV_16SC2)
                image = cv2.remap(image, *undistortion_maps[shape], cv2.INTER_LINEAR)
            else:
                image = cv2.undistort(image, camera, calibration[4:])
        if args.scale != 1:
            image = cv2.resize(image, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        return image[:height-height % 16, :width-width % 16]

    reader = ThreadPoolExecutor(max_workers=1) if args.prefetch else None
    pending = deque()
    if reader:
        pending.extend(reader.submit(read_frame, path) for path in files[:args.prefetch])
    for index, path in enumerate(files):
        start = time.perf_counter()
        if reader:
            image = pending.popleft().result()
            if index + args.prefetch < len(files):
                pending.append(reader.submit(read_frame, files[index + args.prefetch]))
        else:
            image = read_frame(path)
        if slam is None:
            slam = DPVO(cfg, args.network, ht=image.shape[0], wd=image.shape[1])
            if args.trt_encoders:
                from deploy.jetson.tensorrt_encoder import install_encoders
                install_encoders(slam.network, args.trt_encoders, args.network)
            if args.trt_update:
                from deploy.jetson.tensorrt_update import install_update
                install_update(slam.network, args.trt_update, args.network)
            if args.capture_update_inputs:
                def capture_update(module, inputs):
                    if index >= args.warmup and index % 40 == 0 and len(update_samples) < 6 and index not in sampled_frames:
                        sampled_frames.add(index)
                        update_samples.append(tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in inputs))
                slam.network.update.register_forward_pre_hook(capture_update)
            if args.profile:
                update_parts = ([('update_head', slam.network.update.head),
                                 ('update_tail', slam.network.update.tail)] if args.trt_update else
                                [('update_corr', slam.network.update.corr),
                                 ('update_gru', slam.network.update.gru)])
                for name, module in [('fnet', slam.network.patchify.fnet),
                                     ('inet', slam.network.patchify.inet),
                                     ('update', slam.network.update)] + update_parts:
                    events[name], stage_times[name] = [], []

                    def before(module, inputs, name=name):
                        pair = (torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True))
                        pair[0].record()
                        events[name].append(pair)

                    def after(module, inputs, output, name=name):
                        events[name][-1][1].record()

                    module.register_forward_pre_hook(before)
                    module.register_forward_hook(after)
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.tensor(calibration[:4] * args.scale,
                                  dtype=torch.float32, device='cuda')
        torch.cuda.synchronize()
        tick = time.perf_counter()
        slam(index, image, intrinsics)
        torch.cuda.synchronize()
        if ba_errors:
            raise RuntimeError(f'Bundle adjustment failed: {ba_errors[-1]}')
        if slam.is_initialized and initialization_frame is None:
            initialization_frame = index
        times.append(time.perf_counter() - tick)
        for name, pairs in events.items():
            stage_times[name].append(sum(a.elapsed_time(b) for a, b in pairs))
            pairs.clear()
        if index % 25 == 0:
            print(f'frame={index} initialized={slam.is_initialized} '
                  f'keyframes={slam.n} slam_ms={times[-1]*1000:.1f}', flush=True)
        total_times.append(time.perf_counter() - start)
    if reader:
        reader.shutdown(wait=True)
    if not slam.is_initialized:
        raise RuntimeError('DPVO never initialized; timing is not a VO benchmark')
    tick = time.perf_counter()
    poses, stamps = slam.terminate()
    torch.cuda.synchronize()
    if ba_errors:
        raise RuntimeError(f'Final bundle adjustment failed: {ba_errors[-1]}')
    termination = time.perf_counter() - tick
    if not np.isfinite(poses).all():
        raise RuntimeError('Trajectory contains nonfinite values')
    steady_start = max(args.warmup, initialization_frame + 1)
    if steady_start >= len(times):
        raise RuntimeError('No steady-state frames after initialization and warmup')
    steady = np.array(times[steady_start:])
    end_to_end = np.array(total_times[steady_start:])
    result = dict(torch=torch.__version__, cuda=torch.version.cuda,
                  backend='tensorrt-encoders' if args.trt_encoders else 'pytorch',
                  trt_update=args.trt_update,
                  trt_update_fallback_calls=slam.network.update.fallback_calls if args.trt_update else 0,
                  cached_undistortion=args.remap,
                  prefetch=args.prefetch,
                  gpu=torch.cuda.get_device_name(), frames=len(files),
                  warmup=args.warmup, image_hw=list(image.shape[1:]),
                  initialization_frame=initialization_frame,
                  steady_start_frame=steady_start,
                  target_fps=args.target_fps, slam_fps=1/steady.mean(),
                  end_to_end_fps=1/end_to_end.mean(),
                  end_to_end_p95_ms=float(np.percentile(end_to_end, 95)*1000),
                  end_to_end_deadline_miss_fraction=float(
                      (end_to_end > 1/args.target_fps).mean()),
                  slam_p50_ms=float(np.median(steady)*1000),
                  slam_p95_ms=float(np.percentile(steady, 95)*1000),
                  deadline_miss_fraction=float((steady > 1/args.target_fps).mean()),
                  termination_seconds=termination, config=cfg.dump(),
                  stage_mean_ms={name: float(np.mean(values[steady_start:]))
                                 for name, values in stage_times.items()},
                  frame_seconds=times, total_frame_seconds=total_times)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    if args.capture_update_inputs:
        torch.save(update_samples, args.capture_update_inputs)
    np.savetxt(output.with_suffix('.tum'),
               np.column_stack([stamps, poses]), fmt='%.9f')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {'config', 'frame_seconds', 'total_frame_seconds'}}, indent=2))


if __name__ == '__main__':
    main()

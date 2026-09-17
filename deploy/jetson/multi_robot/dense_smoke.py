"""Exercise real DPVO -> TensorRT DA3 -> landmark alignment on a prepared replay.

The NPZ input contains rectified uint8 BGR images [N,240,384] and matching
fx,fy,cx,cy. This does not publish into the live fleet or open a camera.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def main():
    import torch
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    from deploy.jetson.tensorrt_encoder import install_encoders
    from deploy.jetson.dense_mapping import OnlineDenseMapper
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--network', default='/models/dpvo.pth')
    parser.add_argument('--encoders', default='/output/engines-240x384')
    parser.add_argument('--engine', default='/output/da3-small-238x378/da3.engine')
    args = parser.parse_args()
    data = np.load(args.replay)
    images, intrinsics = data['images'], data['intrinsics'].astype(np.float32)
    if images.shape[1:] != (240, 384, 3) or images.dtype != np.uint8 or intrinsics.shape != (4,):
        raise ValueError('Replay must contain BGR uint8 240x384 images and four intrinsics')
    cfg.merge_from_file('/opt/dpvo/config/jetson_online.yaml')
    cfg.CLASSIC_LOOP_CLOSURE = False
    cfg.LOOP_CLOSURE = False
    torch.manual_seed(123)
    np.random.seed(123)
    mapper = OnlineDenseMapper(args.engine)
    slam = None
    accepted = []
    frame_ms = []
    try:
        with torch.inference_mode():
            slam = DPVO(cfg, args.network, ht=240, wd=384)
            install_encoders(slam.network, args.encoders, args.network)
            calibration = torch.from_numpy(intrinsics).cuda()
            cloud = None
            for index, bgr in enumerate(images):
                start = time.monotonic()
                mapper.remember(slam.counter, bgr)
                slam(index * .1, torch.from_numpy(bgr).permute(2, 0, 1).cuda(), calibration)
                value = mapper.update(slam, intrinsics)
                if value is not None and len(value[0]):
                    cloud = value
                    if 'inliers' in mapper.stats and mapper.stats not in accepted:
                        accepted.append(dict(mapper.stats))
                        print(json.dumps(dict(frame=index, points=len(cloud[0]), **mapper.stats)), flush=True)
                torch.cuda.synchronize()
                frame_ms.append((time.monotonic() - start) * 1000)
                if index % 20 == 0:
                    print(json.dumps(dict(frame=index, keyframes=slam.n, initialized=slam.is_initialized,
                                          dense_status=mapper.stats)), flush=True)
                time.sleep(max(0, .1 - (time.monotonic() - start)))
            deadline = time.monotonic() + 5
            while mapper.pending and time.monotonic() < deadline:
                time.sleep(.05)
                value = mapper.update(slam, intrinsics)
                if value is not None and len(value[0]):
                    cloud = value
            report = dict(initialized=slam.is_initialized, keyframes=slam.n,
                          points=0 if cloud is None else len(cloud[0]), accepted=accepted,
                          last_status=mapper.stats, median_frame_ms=float(np.median(frame_ms[10:])))
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'report.json').write_text(json.dumps(report, indent=2))
            if cloud is not None:
                np.savez_compressed(output / 'cloud.npz', points=cloud[0], colors=cloud[1])
            print(json.dumps(report), flush=True)
            if not report['initialized'] or report['points'] < 1000:
                raise RuntimeError('Replay did not produce an aligned dense map')
    finally:
        mapper.close()


if __name__ == '__main__':
    main()

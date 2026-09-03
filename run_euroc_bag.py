import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from rosbags.highlevel import AnyReader

from dpvo.config import cfg
from dpvo.dpvo import DPVO


def decode_image(message):
    data = np.asarray(message.data, dtype=np.uint8)

    if message.encoding in ("mono8", "8UC1"):
        image = data.reshape(message.height, message.step)[:, : message.width]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if message.encoding in ("bgr8", "rgb8"):
        image = data.reshape(message.height, message.step)[:, : message.width * 3]
        image = image.reshape(message.height, message.width, 3)
        if message.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    raise ValueError(f"Unsupported camera encoding: {message.encoding}")


@torch.no_grad()
def run(args):
    calibration = np.loadtxt(args.calib, delimiter=" ")
    fx, fy, cx, cy = calibration[:4]
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsics_np = np.array([fx, fy, cx, cy], dtype=np.float32) * args.scale

    slam = None
    processed = 0
    with AnyReader([args.bag]) as reader:
        connections = [x for x in reader.connections if x.topic == args.topic]
        if not connections:
            raise RuntimeError(f"No {args.topic} topic in {args.bag}")

        total = sum(x.msgcount for x in connections)
        selected = (total + args.stride - 1) // args.stride
        print(
            f"Streaming {args.bag.name}: {total} camera frames, "
            f"stride {args.stride} ({selected} selected)"
        )

        for source_index, (connection, timestamp_ns, rawdata) in enumerate(
            reader.messages(connections=connections)
        ):
            if source_index % args.stride:
                continue

            message = reader.deserialize(rawdata, connection.msgtype)
            image = decode_image(message)
            if len(calibration) > 4:
                image = cv2.undistort(image, camera_matrix, calibration[4:])
            if args.scale != 1.0:
                image = cv2.resize(
                    image,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_AREA,
                )

            height, width = image.shape[:2]
            image = image[: height - height % 16, : width - width % 16]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics_np).cuda()

            if slam is None:
                viewer = args.viewer or (
                    "rerun" if args.rerun_save or args.rerun_connect else None
                )
                slam = DPVO(
                    cfg,
                    args.network,
                    ht=image_tensor.shape[1],
                    wd=image_tensor.shape[2],
                    viewer=viewer,
                    viewer_output=args.rerun_save,
                    viewer_connect=args.rerun_connect,
                )

            slam(timestamp_ns * 1e-9, image_tensor, intrinsics)
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{selected}")
            if args.max_frames is not None and processed >= args.max_frames:
                break

    if slam is None:
        raise RuntimeError(f"No camera frames read from {args.bag}")

    poses, timestamps = slam.terminate()
    points = slam.pg.points_.cpu().numpy()[: slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]
    keyframe_input_indices = slam.pg.tstamps_[: slam.n].astype(np.int64)
    keyframe_timestamps = np.asarray(
        [slam.tlist[int(index)] for index in keyframe_input_indices],
        dtype=np.float64,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            poses=poses,
            timestamps=timestamps,
            points=points,
            colors=colors,
            calibration=calibration,
            session_from_map=slam.pg.session_from_map_,
            keyframe_input_indices=keyframe_input_indices,
            keyframe_timestamps=keyframe_timestamps,
            keyframe_count=np.asarray(slam.n, dtype=np.int64),
            patch_count=np.asarray(slam.m, dtype=np.int64),
            patches_per_keyframe=np.asarray(slam.M, dtype=np.int64),
        )

    print(f"Completed {args.bag.name}: {processed} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--network", default="dpvo.pth")
    parser.add_argument("--config", default="config/fast.yaml")
    parser.add_argument("--calib", default="calib/euroc.txt")
    parser.add_argument("--topic", default="/cam0/image_raw")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--classic-loop-closure", action="store_true")
    parser.add_argument("--classic-pgo-use-threads", action="store_true")
    parser.add_argument("--enable-dpvo-loop-closure", action="store_true")
    parser.add_argument("--max-edge-age", type=int, default=1000)
    parser.add_argument("--loop-retr-thresh", type=float, default=0.04)
    parser.add_argument("--orb-vocab", type=Path)
    parser.add_argument("--viewer", choices=["pangolin", "rerun"])
    parser.add_argument("--rerun-save", type=Path)
    parser.add_argument("--rerun-connect")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.rerun_save and args.rerun_connect:
        parser.error("--rerun-save and --rerun-connect are mutually exclusive")
    if (args.rerun_save or args.rerun_connect) and args.viewer == "pangolin":
        parser.error("Rerun output options cannot be combined with Pangolin")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if not 0.0 < args.scale <= 1.0:
        parser.error("--scale must be in (0, 1]")

    cfg.merge_from_file(args.config)
    cfg.LOOP_CLOSURE = args.enable_dpvo_loop_closure
    cfg.MAX_EDGE_AGE = args.max_edge_age
    cfg.LOOP_RETR_THRESH = args.loop_retr_thresh
    if args.classic_loop_closure:
        cfg.CLASSIC_LOOP_CLOSURE = True
        cfg.CLASSIC_PGO_USE_THREADS = args.classic_pgo_use_threads
    if args.orb_vocab:
        cfg.ORB_VOCAB_PATH = str(args.orb_vocab)
    torch.manual_seed(1234)
    print(cfg)
    run(args)


if __name__ == "__main__":
    main()

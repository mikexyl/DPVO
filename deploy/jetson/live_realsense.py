"""Live RealSense DPVO with bounded capture latency and the existing Rerun viewer."""
import argparse
import json
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from dpvo.config import cfg
from tensorrt_encoder import install_encoders
from realsense_camera import LatestFrame, camera_maps, color_to_bgr
from tracking_health import tracking_health


@torch.no_grad()
def main(args=None, viewer_factory=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--network', default='/models/dpvo.pth')
    parser.add_argument('--config', default='config/jetson_live.yaml')
    parser.add_argument('--trt-encoders', default='/output/engines-240x384')
    parser.add_argument('--web-port', type=int, default=9090)
    parser.add_argument('--viewer', choices=['rerun', 'viser'], default='rerun')
    parser.add_argument('--start-paused', action='store_true',
                        help='Serve the Viser control panel without starting tracking')
    parser.add_argument('--grpc-port', type=int, default=9876)
    parser.add_argument('--web-origin', action='append', default=[])
    parser.add_argument('--viewer-fps', type=float, default=5)
    parser.add_argument('--camera-fps', type=int, default=30)
    parser.add_argument('--camera-format', choices=['bgr8', 'yuyv'], default='bgr8',
                        help='Use raw YUYV with CPU conversion for JetPack 6.2')
    parser.add_argument('--stride', type=int, default=3,
                        help='Select every Nth raw color frame before copying or processing')
    parser.add_argument('--serial')
    parser.add_argument('--dense-engine', default='/output/da3-small-238x378/da3.engine')
    parser.add_argument('--dense-fps', type=float, default=.5)
    parser.add_argument('--output', default='/output/live')
    parser.add_argument('--max-frames', type=int, default=0, help='Zero runs until stopped')
    parser.add_argument('--record-frames', type=int, default=0,
                        help='Save a bounded diagnostic sequence of rectified inputs')
    args = parser.parse_args() if args is None else args
    if args.start_paused and args.viewer != 'viser':
        parser.error('--start-paused requires the Viser control panel')
    if args.viewer_fps <= 0 or args.max_frames < 0 or args.stride < 1:
        parser.error('viewer-fps and stride must be positive; max-frames nonnegative')
    if args.viewer == 'viser' and viewer_factory is None:
        from viser_control import run_control_panel
        return run_control_panel(args)
    # DPVO initializes a CUDA identity tensor at import time. Keep that import
    # in the worker so the idle web process never creates a CUDA context.
    from dpvo import fastba
    from dpvo.dpvo import DPVO
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    cfg.merge_from_file(args.config)
    slam = DPVO(cfg, args.network, ht=240, wd=384)
    if args.trt_encoders:
        install_encoders(slam.network, args.trt_encoders, args.network)
    if args.viewer == 'viser':
        viewer = viewer_factory(slam.pg, 240, 384, web_port=args.web_port)
    else:
        import rerun as rr
        from dpvo.rerun_viewer import RerunViewer
        viewer = RerunViewer(slam.pg, 240, 384, web_port=args.web_port,
                             grpc_port=args.grpc_port, cors_allow_origin=args.web_origin)

    def viewer_status(text):
        if args.viewer == 'viser':
            viewer.set_status(text)
        else:
            rr.log('status', rr.TextDocument(text))

    def reset_viewer():
        viewer.num_frames = viewer.num_points = 0
        if args.viewer == 'viser':
            viewer.reset()
        else:
            rr.log('world', rr.Clear(recursive=True))
    pipeline = rs.pipeline()
    configuration = rs.config()
    if args.serial:
        configuration.enable_device(args.serial)
    configuration.enable_stream(rs.stream.color, 1280, 800, getattr(rs.format, args.camera_format), args.camera_fps)
    profile = pipeline.start(configuration)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    maps, output_intrinsics = camera_maps(intr, 384, 240)
    calibration = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                       cx=intr.ppx, cy=intr.ppy, distortion=str(intr.model), coeffs=intr.coeffs,
                       nominal_fov_degrees=rs.rs2_fov(intr), crop_left=0,
                       output_intrinsics=output_intrinsics, output_width=384, output_height=240)
    (output / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    recording = output / f'diagnostic-{time.time_ns()}' if args.record_frames else None
    if recording:
        (recording / 'images').mkdir(parents=True)
        np.savetxt(recording / 'calib.txt', [output_intrinsics])
    intrinsics = torch.tensor(output_intrinsics,
                              device='cuda', dtype=torch.float32)
    capture = LatestFrame(pipeline, stride=args.stride)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    ba_errors = []
    original_ba = fastba.BA

    def checked_ba(*positional, **keywords):
        try:
            return original_ba(*positional, **keywords)
        except Exception as error:
            ba_errors.append(str(error))
            raise

    fastba.BA = checked_ba
    previous = processed = session = session_frames = resets = 0
    started = last_report = time.perf_counter()
    last_view = 0
    report_count = 0
    first_timestamp = None
    last_loss = None
    status = {}
    dense_mapper = None
    dense_error = None
    def reset_dense():
        nonlocal dense_mapper
        if dense_mapper is not None:
            dense_mapper.close()
            dense_mapper = None
        if getattr(args, 'enable_dense_mapping', False):
            from deploy.jetson.dense_mapping import OnlineDenseMapper
            dense_mapper = OnlineDenseMapper(args.dense_engine, fps=args.dense_fps)
    try:
        reset_dense()
        while not stop.is_set() and (not args.max_frames or processed < args.max_frames):
            number, timestamp, arrival, image = capture.get(previous)
            previous = number
            if first_timestamp is None:
                first_timestamp = timestamp
            image = color_to_bgr(image, args.camera_format)
            image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
            if recording and processed < args.record_frames:
                cv2.imwrite(str(recording / 'images' / f'{timestamp:.6f}.png'), image,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if dense_mapper is not None:
                dense_mapper.remember(int(slam.counter), image)
            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            begin = time.perf_counter()
            slam(timestamp - first_timestamp, image, intrinsics)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            processed += 1
            session_frames += 1
            health = tracking_health(slam.pg.poses_[:slam.n],
                                     slam.pg.patches_[max(0, slam.n-3):slam.n, :, 2],
                                     slam.is_initialized)
            finite, median_inverse_depth = health['finite'], health['median_inverse_depth']
            if ba_errors or health['lost']:
                last_loss = dict(frame=processed, median_inverse_depth=median_inverse_depth,
                                 finite_poses=finite, ba_errors=list(ba_errors))
                print(json.dumps(dict(state='tracking_lost', **last_loss)), flush=True)
                viewer_status('Tracking lost; restarting initialization.')
                reset_viewer()
                reset_dense()
                network = slam.network
                del slam
                slam = DPVO(cfg, network, ht=240, wd=384)
                viewer.patch_graph = slam.pg
                viewer.num_frames = viewer.num_points = 0
                session += 1
                resets += 1
                session_frames = 0
                ba_errors.clear()
                continue
            if dense_mapper is not None:
                try:
                    cloud = dense_mapper.update(slam, np.asarray(output_intrinsics, dtype=np.float32))
                    if cloud is not None:
                        viewer.update_dense(*cloud)
                except Exception as error:
                    dense_error = str(error)
                    dense_mapper.close()
                    dense_mapper = None
            if finished - last_view >= 1 / args.viewer_fps:
                if args.viewer == 'rerun':
                    rr.set_time('capture_time', duration=timestamp - first_timestamp)
                viewer.update_image(image)
                viewer.update_state(intrinsics, slam.n if slam.is_initialized else 0,
                                    slam.m if slam.is_initialized else 0)
                text = ('Tracking (monocular scale)' if slam.is_initialized else
                        'Waiting for motion: move the camera slowly sideways to initialize.')
                if dense_error:
                    text += f' Dense mapping stopped: {dense_error}'
                elif dense_mapper is not None:
                    count = sum(len(points) for points, _ in dense_mapper.cloud.frames.values())
                    text += f' Dense mapping: {count} points.'
                    if 'rejected' in dense_mapper.stats:
                        text += ' Waiting for reliable landmark alignment.'
                viewer_status(text)
                last_view = finished
            if finished - last_report >= 1:
                eligible_frames = (number - 1) // args.stride + 1
                status = dict(state='tracking' if slam.is_initialized else 'waiting_for_motion',
                              tracking_enabled=True, camera_running=True,
                              viewer=args.viewer, dense_enabled=dense_mapper is not None,
                              dense_status=dense_mapper.stats if dense_mapper else dense_error,
                              processed_frames=processed,
                              captured_frames=number, skipped_frames=number-processed, session=session,
                              stride=args.stride, selected_input_fps=args.camera_fps/args.stride,
                              stride_skipped_frames=number-eligible_frames,
                              overload_skipped_frames=eligible_frames-processed,
                              keyframes=slam.n, points=slam.m if slam.is_initialized else 0,
                              processing_fps=(processed-report_count)/(finished-last_report),
                              inference_ms=1000*(finished-begin),
                              arrival_to_pose_ms=1000*(finished-arrival),
                              pose=slam.pg.poses_[slam.n-1].detach().cpu().tolist(),
                              median_inverse_depth=median_inverse_depth, last_tracking_loss=last_loss,
                              uptime_seconds=finished-started, resets=resets)
                temporary = output / 'status.tmp'
                temporary.write_text(json.dumps(status, indent=2))
                temporary.replace(output / 'status.json')
                print(json.dumps(status), flush=True)
                last_report, report_count = finished, processed
            # Bound both GPU graph storage and CPU timestamp/pose history.
            if slam.n >= slam.N - 2 or session_frames >= 8192:
                network = slam.network
                del slam
                slam = DPVO(cfg, network, ht=240, wd=384)
                viewer.patch_graph = slam.pg
                reset_viewer()
                reset_dense()
                session += 1
                resets += 1
                session_frames = 0
    except Exception as error:
        status.update(state='error', error=repr(error))
        raise
    finally:
        if dense_mapper is not None:
            dense_mapper.close()
        capture.stop.set()
        capture.thread.join(timeout=6)
        pipeline.stop()
        viewer.join()
        fastba.BA = original_ba
        status.setdefault('state', 'stopped')
        if status['state'] != 'error':
            status['state'] = 'stopped'
        (output / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()

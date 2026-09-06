import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(
    cfg,
    network,
    imagedir,
    calib,
    stride=1,
    skip=0,
    viz=False,
    timeit=False,
    viewer=None,
    viewer_output=None,
    viewer_connect=None,
    yolo_model=None,
    yolo_confidence=0.25,
    yolo_image_size=640,
    yolo_task=None,
    scene_graph=False,
    scene_graph_output=None,
    da3_engine=None,
    dense_map_output=None,
    dense_map_stride=7,
    dense_map_max_error=0.25,
    sam_model=None,
    sam_points_per_side=32,
    sam_points_per_batch=16,
    sam_min_mask_area=100,
    sam_pred_iou_thresh=0.8,
    sam_stability_thresh=0.92,
    sam_output=None,
    sam_video=False,
    sam_video_max_tracks=8,
    sam_video_refresh=10,
    sam_video_memory=3,
    sphere_options=None,
):

    slam = None
    queue = Queue(maxsize=8)

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(
                cfg,
                network,
                ht=H,
                wd=W,
                viz=viz,
                viewer=viewer,
                viewer_output=viewer_output,
                viewer_connect=viewer_connect,
                yolo_model=yolo_model,
                yolo_confidence=yolo_confidence,
                yolo_image_size=yolo_image_size,
                yolo_task=yolo_task,
                scene_graph=scene_graph,
                scene_graph_output=scene_graph_output,
                da3_engine=da3_engine,
                dense_map_output=dense_map_output,
                dense_map_stride=dense_map_stride,
                dense_map_max_error=dense_map_max_error,
                sam_model=sam_model,
                sam_points_per_side=sam_points_per_side,
                sam_points_per_batch=sam_points_per_batch,
                sam_min_mask_area=sam_min_mask_area,
                sam_pred_iou_thresh=sam_pred_iou_thresh,
                sam_stability_thresh=sam_stability_thresh,
                sam_output=sam_output,
                sam_video=sam_video,
                sam_video_max_tracks=sam_video_max_tracks,
                sam_video_refresh=sam_video_refresh,
                sam_video_memory=sam_video_memory,
                sphere_options=sphere_options,
            )

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        '--viz',
        action="store_true",
        help="visualize with the legacy Pangolin viewer",
    )
    viewer_group.add_argument(
        '--viewer',
        choices=["pangolin", "rerun"],
        help="select a live visualization backend",
    )
    parser.add_argument(
        '--rerun-save',
        metavar="PATH",
        help="save Rerun data to an .rrd file instead of spawning the viewer",
    )
    parser.add_argument(
        '--rerun-connect',
        metavar="URL",
        help="stream to a Rerun viewer, e.g. rerun+http://host:9876/proxy",
    )
    segmentation_group = parser.add_mutually_exclusive_group()
    segmentation_group.add_argument(
        '--yolo-model',
        metavar="PATH",
        help="run a TensorRT YOLO .engine model and overlay detections in Rerun",
    )
    parser.add_argument('--yolo-confidence', type=float, default=0.25)
    parser.add_argument('--yolo-image-size', type=int, default=640)
    parser.add_argument('--yolo-task', choices=['detect', 'segment'])
    segmentation_group.add_argument(
        '--sam-model', metavar='PATH',
        help='SAM 2.1 Tiny TensorRT bundle .json or reference PyTorch .pt checkpoint',
    )
    parser.add_argument('--sam-points-per-side', type=int, default=32,
                        help='automatic prompt grid width/height (default: 32)')
    parser.add_argument('--sam-points-per-batch', type=int, default=16,
                        help='PyTorch prompt batch size (default: 16); TensorRT uses its exported batch')
    parser.add_argument('--sam-min-mask-area', type=int, default=100,
                        help='minimum region area in original DPVO image pixels')
    parser.add_argument('--sam-pred-iou-thresh', type=float, default=0.8,
                        help='minimum predicted mask IoU, not class confidence (default: 0.8)')
    parser.add_argument('--sam-stability-thresh', type=float, default=0.92,
                        help='minimum mask stability score (default: 0.92)')
    parser.add_argument('--sam-output', metavar='PATH',
                        help='save per-frame SAM statistics and a final mask preview')
    parser.add_argument('--sam-video', action='store_true',
                        help='enable SAM temporal-memory tracking (TRT encoder + BF16 PyTorch video modules)')
    parser.add_argument('--sam-video-max-tracks', type=int, default=8,
                        help='maximum automatically seeded video regions (default: 8)')
    parser.add_argument('--sam-video-refresh', type=int, default=10,
                        help='discover/re-prompt every N processed frames; 0 seeds once (default: 10)')
    parser.add_argument('--sam-video-memory', type=int, default=3,
                        help='memory slots including the prompt frame, 2-7 (default: 3; original SAM: 7)')
    parser.add_argument(
        '--scene-graph',
        action='store_true',
        help='fuse YOLO regions with DPVO patches into persistent 3D object nodes',
    )
    parser.add_argument(
        '--scene-graph-output',
        metavar='PATH',
        help='write the scene graph as JSON (also enables --scene-graph)',
    )
    parser.add_argument(
        '--da3-engine',
        metavar='PATH',
        help='build an aligned dense map with a fixed two-view DA3 TensorRT engine',
    )
    parser.add_argument(
        '--dense-map-output',
        metavar='PATH',
        help='write the aligned dense map as binary PLY',
    )
    parser.add_argument(
        '--dense-map-stride',
        type=int,
        default=7,
        help='sample every Nth DA3 depth pixel for the dense map (default: 7)',
    )
    parser.add_argument(
        '--dense-map-max-error',
        type=float,
        default=0.25,
        help='reject keyframe pairs above this median relative alignment error',
    )
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--local-spheres', action='store_true',
                        help='periodically project a sliding DA3 keyframe map at its newest camera center')
    parser.add_argument('--sphere-window', type=int, default=30,
                        help='recent accepted dense keyframes in the local map, 2-60 (default: 30)')
    parser.add_argument('--sphere-every', type=int, default=5,
                        help='render after N new accepted dense keyframes (default: 5)')
    parser.add_argument('--sphere-width', type=int, default=1024,
                        help='equirectangular width, divisible by 4, 256-2048 (default: 1024)')
    parser.add_argument('--sphere-radius', type=float,
                        help='display radius in DPVO units; default: 0.2 times median radial depth')
    parser.add_argument('--sphere-output-dir', metavar='PATH',
                        help='new output directory for local sphere PNG/NPZ snapshots')
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--save_colmap', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    if args.rerun_save and args.rerun_connect:
        parser.error("--rerun-save and --rerun-connect are mutually exclusive")
    if (args.rerun_save or args.rerun_connect) and (
        args.viz or args.viewer == "pangolin"
    ):
        parser.error("Rerun output options cannot be combined with Pangolin")

    viewer = args.viewer or (
        "rerun"
        if args.rerun_save or args.rerun_connect or args.yolo_model or args.sam_model or args.da3_engine
        else None
    )
    if args.yolo_model and viewer != "rerun":
        parser.error("--yolo-model requires the Rerun viewer")
    if args.sam_model and (args.viz or viewer != "rerun"):
        parser.error("--sam-model requires the Rerun viewer")
    if args.sam_model:
        sam_path = Path(args.sam_model)
        if sam_path.suffix != ".json" and sam_path.name not in {"sam2.1_hiera_tiny.pt", "sam2.1_t.pt"}:
            parser.error("--sam-model must be a TensorRT bundle .json or SAM 2.1 Tiny checkpoint")
        if not sam_path.is_file():
            parser.error(f"SAM checkpoint not found: {sam_path}")
    if args.sam_output and not args.sam_model:
        parser.error("--sam-output requires --sam-model")
    if args.sam_video and (not args.sam_model or Path(args.sam_model).suffix != '.json'):
        parser.error("--sam-video requires a TensorRT SAM bundle .json for image features and automatic seeding")
    if args.sam_video_max_tracks < 1 or args.sam_video_refresh < 0 or not 2 <= args.sam_video_memory <= 7:
        parser.error("SAM video requires positive track count, nonnegative refresh interval and 2-7 memories")
    if args.sam_points_per_side < 1 or args.sam_points_per_batch < 1 or args.sam_min_mask_area < 0:
        parser.error("SAM grid/batch sizes must be positive and minimum mask area nonnegative")
    if not 0 <= args.sam_pred_iou_thresh <= 1 or not 0 <= args.sam_stability_thresh <= 1:
        parser.error("SAM quality thresholds must be between zero and one")
    if args.da3_engine and viewer != "rerun":
        parser.error("--da3-engine requires the Rerun viewer")
    if args.dense_map_output and not args.da3_engine:
        parser.error("--dense-map-output requires --da3-engine")
    sphere_options = None
    if args.local_spheres:
        if not args.da3_engine or args.viz or viewer != 'rerun':
            parser.error("--local-spheres requires --da3-engine and the Rerun viewer")
        if not 2 <= args.sphere_window <= 60 or args.sphere_every < 1:
            parser.error("local spheres require a 2-60 keyframe window and positive render interval")
        if not 256 <= args.sphere_width <= 2048 or args.sphere_width % 4:
            parser.error("--sphere-width must be divisible by 4 and between 256 and 2048")
        if args.sphere_radius is not None and (not np.isfinite(args.sphere_radius) or args.sphere_radius <= 0):
            parser.error("--sphere-radius must be finite and positive")
        sphere_options = dict(window_size=args.sphere_window, every=args.sphere_every,
                              width=args.sphere_width, radius=args.sphere_radius,
                              output_dir=args.sphere_output_dir or f"saved_spheres/{args.name}")
        if (Path(sphere_options['output_dir']) / 'manifest.json').exists():
            parser.error("use a new --sphere-output-dir or --name to preserve existing snapshots")
    elif args.sphere_output_dir:
        parser.error("--sphere-output-dir requires --local-spheres")
    scene_graph = args.scene_graph or args.scene_graph_output is not None
    if scene_graph and not args.yolo_model:
        parser.error("--scene-graph requires --yolo-model")
    scene_graph_output = args.scene_graph_output
    if scene_graph and scene_graph_output is None:
        scene_graph_output = f"saved_scene_graphs/{args.name}.json"
    dense_map_output = args.dense_map_output
    if args.da3_engine and dense_map_output is None:
        dense_map_output = f"saved_dense_maps/{args.name}.ply"
    sam_output = args.sam_output
    if args.sam_model and sam_output is None:
        sam_output = f"saved_segmentations/{args.name}.json"

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    (poses, tstamps), (points, colors, calib) = run(
        cfg,
        args.network,
        args.imagedir,
        args.calib,
        stride=args.stride,
        skip=args.skip,
        viz=args.viz,
        timeit=args.timeit,
        viewer=viewer,
        viewer_output=args.rerun_save,
        viewer_connect=args.rerun_connect,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        yolo_task=args.yolo_task,
        scene_graph=scene_graph,
        scene_graph_output=scene_graph_output,
        da3_engine=args.da3_engine,
        dense_map_output=dense_map_output,
        dense_map_stride=args.dense_map_stride,
        dense_map_max_error=args.dense_map_max_error,
        sam_model=args.sam_model,
        sam_points_per_side=args.sam_points_per_side,
        sam_points_per_batch=args.sam_points_per_batch,
        sam_min_mask_area=args.sam_min_mask_area,
        sam_pred_iou_thresh=args.sam_pred_iou_thresh,
        sam_stability_thresh=args.sam_stability_thresh,
        sam_output=sam_output,
        sam_video=args.sam_video,
        sam_video_max_tracks=args.sam_video_max_tracks,
        sam_video_refresh=args.sam_video_refresh,
        sam_video_memory=args.sam_video_memory,
        sphere_options=sphere_options,
    )
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    if args.save_ply:
        save_ply(args.name, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {args.name}", filename=f"trajectory_plots/{args.name}.pdf")


        

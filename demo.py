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
    parser.add_argument(
        '--yolo-model',
        metavar="PATH",
        help="run a TensorRT YOLO .engine model and overlay detections in Rerun",
    )
    parser.add_argument('--yolo-confidence', type=float, default=0.25)
    parser.add_argument('--yolo-image-size', type=int, default=640)
    parser.add_argument('--yolo-task', choices=['detect', 'segment'])
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
        if args.rerun_save or args.rerun_connect or args.yolo_model or args.da3_engine
        else None
    )
    if args.yolo_model and viewer != "rerun":
        parser.error("--yolo-model requires the Rerun viewer")
    if args.da3_engine and viewer != "rerun":
        parser.error("--da3-engine requires the Rerun viewer")
    if args.dense_map_output and not args.da3_engine:
        parser.error("--dense-map-output requires --da3-engine")
    scene_graph = args.scene_graph or args.scene_graph_output is not None
    if scene_graph and not args.yolo_model:
        parser.error("--scene-graph requires --yolo-model")
    scene_graph_output = args.scene_graph_output
    if scene_graph and scene_graph_output is None:
        scene_graph_output = f"saved_scene_graphs/{args.name}.json"
    dense_map_output = args.dense_map_output
    if args.da3_engine and dense_map_output is None:
        dense_map_output = f"saved_dense_maps/{args.name}.ply"

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


        

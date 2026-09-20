"""Fresh, causally recorded multi-robot replay; render its snapshots to MP4.

Reuses the ROS DPVO/TUM players and OnlineCbsNode. All trackers share one CUDA
process and a single-threaded ROS executor; CBS runs concurrently in its existing
worker/subprocess. This is co-hosted virtual-robot replay, not a real-time or
physically distributed hardware benchmark. No saved graph or alignment is input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2/dpvo_multi_robot"))
from dpvo.loop_closure.centralized import Sim3
from dpvo.map_gauge import transform_points, transform_poses_xyzw


def warp_live_map(points, owners, poses, timestamps, optimized, anchor):
    """Use only completed CBS poses; grow unseen keyframes using its map anchor.

    Correspondence is source timestamp, never a mutable DPVO keyframe index.
    Each landmark stays attached to the keyframe that generated its patch.
    """
    transforms = [optimized.get(round(float(t), 6),
                  anchor.compose(Sim3.from_pose(p)))
                  for p, t in zip(poses, timestamps)]
    warped = np.empty_like(points)
    for index in np.unique(owners):
        mask = owners == index
        delta = transforms[index].compose(Sim3.from_pose(poses[index]).inverse())
        warped[mask] = delta.apply(points[mask])
    trajectory = np.array([p.translation for p in transforms]).reshape(-1, 3)
    return warped, trajectory


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def display_cloud(points):
    """Clip extreme sparse-depth rays for display only, in the current frame.

    The original snapshots and every trajectory position remain untouched.
    Median + 10 MAD is computed independently for each robot's current cloud;
    no final-map extent, future frame, or ground truth enters this filter.
    """
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 10:
        return points
    radius = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    median = np.median(radius)
    mad = np.median(np.abs(radius - median))
    return points[radius <= median + 10*max(mad, 1e-6)]


def map_viewports(groups):
    """Give an established shared map room, retaining separate unaligned panels."""
    largest = max(range(len(groups)), key=lambda i:len(groups[i][1]))
    total = sum(len(members) for _, members in groups)
    if len(groups) > 1 and len(groups[largest][1]) >= max(3, total/3):
        result = [(groups[largest], (22, 134, 980, 850))]
        rest = [g for i, g in enumerate(groups) if i != largest]
        cols = 1 if len(rest) <= 4 else 2
        rows = math.ceil(len(rest)/cols)
        for i, group in enumerate(rest):
            result.append((group, (1002+(i%cols)*(420//cols),
                                   134+(i//cols)*(850//rows), 420//cols, 850//rows)))
        return result
    cols = min(len(groups), max(1, math.ceil(math.sqrt(len(groups)*1.45))))
    rows = math.ceil(len(groups)/cols)
    return [(group, (22+(i%cols)*(1400//cols), 134+(i//cols)*(850//rows),
                     1400//cols, 850//rows)) for i, group in enumerate(groups)]


def run(args):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from dpvo.lietorch import SE3
    from dpvo_multi_robot.node import MultiRobotDpvoNode
    from dpvo_multi_robot.tum_player import TumRgbPlayer
    from dpvo_multi_robot.online_cbs import OnlineCbsNode
    from dpvo_multi_robot.online_common import robot_components
    from dpvo_multi_robot.cbs_pgo import _read_trajectory
    import torch

    manifest = json.loads(args.manifest.read_text())
    robots = manifest["robots"][:args.robot_count]
    if not robots or len({r["robot_id"] for r in robots}) != len(robots):
        raise ValueError("Expected unique robot IDs")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "snapshots").mkdir()
    (args.output / "previews").mkdir()
    origin = time.monotonic()
    journal = (args.output / "events.jsonl").open("w", buffering=1)
    def event(kind, **value):
        journal.write(json.dumps(dict(elapsed=time.monotonic()-origin,
                                      event=kind, **value), allow_nan=False) + "\n")
    def parameters(values):
        return [Parameter(k, value=v) for k, v in values.items()]

    class RecordingCbs(OnlineCbsNode):
        def __init__(self, **kwargs):
            self.completed = 0
            self.optimized = {}
            self.latest_completion = 0.
            self.failures = []
            super().__init__(**kwargs)

        def _run(self, constraints, paths, sessions, online, timestamps=None):
            event("cbs_cycle_started", robots=sorted(paths),
                  poses=sum(map(len, paths.values())), loops=len(constraints))
            before = self.completed
            members, eligible = self._eligible(constraints, paths, sessions)
            expected = sum(len(group) > 1 for group in robot_components(members,
                [(c.query_robot[0], c.match_robot[0]) for c in eligible]))
            super()._run(constraints, paths, sessions, online, timestamps)
            if self.completed - before != expected:
                self.failures.append(dict(elapsed=time.monotonic()-origin,
                                          reason="cycle did not commit every connected component"))
                event("cbs_cycle_failed", reason=self.failures[-1]["reason"])

        def _command(self, graph_path, output_dir):
            command = super()._command(graph_path, output_dir)
            write_json(output_dir / "command.json", command)
            return command

        def _publish_results(self, graph, output_dir, online, duration):
            with self.lock:
                super()._publish_results(graph, output_dir, online, duration)
                if any(self.active_sessions.get(v.robot_id) != v.session_id
                       for v in graph.vertices):
                    return
                estimates = _read_trajectory(output_dir / "cbs.csv")
                grouped = {}
                for v in graph.vertices:
                    grouped.setdefault(v.robot_id, {})[round(float(v.timestamp), 6)] = estimates[
                        (v.robot_id, v.session_id, v.keyframe_id)]
                self.optimized.update(grouped)
                self.completed += 1
                self.latest_completion = time.monotonic()-origin
                event("cbs_cycle_completed", cycle=self.completed,
                      output_dir=str(output_dir), robots=sorted(grouped),
                      poses=len(graph.vertices), loops=len(graph.edges)-len(graph.vertices)+len(grouped),
                      duration=duration)

    rclpy.init(args=[])
    executor = SingleThreadedExecutor()
    trackers, players = [], []
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    cbs = None
    report = dict(status="running", protocol="fresh_online_virtual_robot_replay",
                  source_manifest=manifest, command=sys.argv,
                  pose_rounds=50, anchor_rounds=50, target_hellinger=.1,
                  d_reset=1.1, original_measurement_weights=True,
                  recording_period_seconds=args.record_period,
                  cbs_period_seconds=args.cbs_period, no_ground_truth=True,
                  precomputed_tracking_or_loops_used=False,
                  scheduling="single CUDA process, interleaved ROS callbacks; concurrent CPU CBS",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output / "run.json", report)
    try:
        cbs = RecordingCbs(parameter_overrides=parameters(dict(
            robot_ids=[r["robot_id"] for r in robots],
            output_dir=str(args.output / "cbs"), archive_snapshots=True,
            cbs_executable=manifest["cbs_executable"], online_period=args.cbs_period,
            iterations=100, stage_mode="alternating", pose_block_iterations=50,
            anchor_block_iterations=50, target_hellinger=.1, d_reset=1.1,
            run_centralized_baseline=False, run_explicit_anchor_centralized_baseline=False,
            timeout_seconds=180., pose_graph_odometry_weight=100.)))
        executor.add_node(cbs)
        for spec in robots:
            robot = spec["robot_id"]
            values = dict(robot_id=robot, session_id=f"video_{robot}",
                          network=manifest["network"], config=manifest["config"],
                          orb_vocab=manifest["orb_vocab"], image_scale=.5,
                          image_topic="camera/image_raw", camera_info_topic="camera/camera_info",
                          session_frame_ids=True, random_seed=1234,
                          retrieval_backend="dbow2", local_feature_backend="disk",
                          bow_threshold=.01, bow_repetitions=1, bow_nms_radius=10,
                          bow_backfill=True, teaser_noise_bound=.1, min_inliers=18,
                          min_inlier_ratio=.15, max_depth=20., teaser_required=True,
                          max_edge_age=48, online_preview_fps=0.,
                          loop_diagnostics_output=str(args.output / f"{robot}_loops.json"),
                          loop_diagnostics_period=10., exit_on_player_done=False,
                          tracking_artifact_output=str(args.output / "tracking"))
            node = MultiRobotDpvoNode(namespace=robot, parameter_overrides=parameters(values))
            trackers.append(node)
            executor.add_node(node)
            player = TumRgbPlayer(namespace=robot, parameter_overrides=parameters(dict(
                sequence_dir=spec["sequence_dir"], stride=1,
                max_frames=args.max_frames, crop_x=0, crop_y=0,
                fx=727.10, fy=727.10, cx=960., cy=540., k1=.00044,
                k2=0., p1=0., p2=0., k3=0., exit_on_finish=False)))
            players.append(player)
            executor.add_node(player)
        event("all_robots_ready", count=len(trackers))
        last_record, done_at, frame, last_cycles = -math.inf, None, 0, -1
        while rclpy.ok() and not stopping:
            executor.spin_once(timeout_sec=.005)
            elapsed = time.monotonic()-origin
            if elapsed > args.timeout:
                raise TimeoutError("Online replay exceeded hard timeout")
            if cbs.failures:
                raise RuntimeError("A CBS cycle failed; see events.jsonl (no automatic retry)")
            all_done = all(p.finished for p in players)
            if all_done and done_at is None:
                done_at = elapsed
                event("all_inputs_finished")
            if elapsed-last_record >= args.record_period or cbs.completed != last_cycles:
                with cbs.lock:
                    packet, arrays = dict(elapsed=time.monotonic()-origin, cycle=cbs.completed,
                                          cbs_running=bool(cbs.worker and cbs.worker.is_alive()),
                                          loops=len(cbs.constraints), robots={}), {}
                    alignments = dict(cbs.alignments)
                    solutions = dict(cbs.optimized)
                for node, player in zip(trackers, players):
                    robot, slam = node.robot_id, node.slam
                    info = dict(frames=0, keyframes=0, finished=player.finished,
                                expected_frames=len(player.selected_frames),
                                group=[robot], image=None)
                    packet["robots"][robot] = info
                    if slam is None or slam.n == 0:
                        continue
                    info.update(frames=int(slam.counter), keyframes=int(slam.n),
                                source_time=float(slam.tlist[-1]), initialized=bool(slam.is_initialized))
                    poses = SE3(slam.pg.poses_[:slam.n]).inv().data.detach().cpu().numpy()
                    poses = transform_poses_xyzw(slam.pg.session_from_map_, poses)
                    stamps = np.array([slam.tlist[int(i)] for i in slam.pg.tstamps_[:slam.n]])
                    step = max(1, math.ceil(slam.m/args.max_points))
                    indices = np.arange(0, slam.m, step)
                    points = slam.pg.points_[indices].detach().cpu().numpy()
                    points = transform_points(slam.pg.session_from_map_, points)
                    colors = slam.pg.colors_.reshape(-1, 3)[indices].detach().cpu().numpy()
                    valid = np.isfinite(points).all(axis=1)
                    owners = indices[valid]//slam.M
                    points, colors = points[valid], colors[valid]
                    alignment = alignments.get(robot)
                    anchor = Sim3.identity()
                    if alignment:
                        from scipy.spatial.transform import Rotation
                        anchor = Sim3(alignment["translation"],
                                      Rotation.from_quat(alignment["quaternion"]).as_matrix(), alignment["scale"])
                        info["group"] = sorted(alignment["sessions"])
                    points, path = warp_live_map(points, owners, poses, stamps,
                                                 solutions.get(robot, {}), anchor)
                    arrays[robot+"_points"] = points.astype(np.float32)
                    arrays[robot+"_colors"] = colors
                    arrays[robot+"_path"] = path.astype(np.float32)
                    # The saved input image is the exact most recently processed frame.
                    image_path = node.tracking_artifact_dir / "frames" / f"{slam.counter-1:08d}.jpg"
                    preview = cv2.imread(str(image_path))
                    if preview is not None:
                        preview = cv2.resize(preview, (384, 216), interpolation=cv2.INTER_AREA)
                        name = f"{frame:06d}_{robot}.jpg"
                        if not cv2.imwrite(str(args.output / "previews" / name), preview):
                            raise RuntimeError("Could not write preview")
                        info["image"] = "previews/"+name
                np.savez_compressed(args.output / "snapshots" / f"{frame:06d}.npz", **arrays)
                write_json(args.output / "snapshots" / f"{frame:06d}.json", packet)
                event("snapshot", index=frame, cycle=packet["cycle"],
                      frames=sum(v["frames"] for v in packet["robots"].values()))
                print(f"ONLINE {elapsed:.1f}s snapshot={frame} cycles={cbs.completed} loops={len(cbs.constraints)} "
                      f"frames={[v['frames'] for v in packet['robots'].values()]} "
                      f"gpu_allocated_GB={torch.cuda.memory_allocated()/1e9:.2f}", flush=True)
                frame += 1
                last_record, last_cycles = elapsed, packet["cycle"]
            if done_at is not None and elapsed-done_at >= args.settle_seconds:
                if not cbs.worker or not cbs.worker.is_alive():
                    break
        if stopping:
            raise RuntimeError("Replay stopped before completion")
        counts = {n.robot_id:int(n.slam.counter) if n.slam else 0 for n in trackers}
        if any(counts[n.robot_id] != len(p.selected_frames) for n,p in zip(trackers,players)):
            raise RuntimeError("Input acknowledgement/count mismatch")
        report.update(status="complete", elapsed_seconds=time.monotonic()-origin,
                      completed_cbs_cycles=cbs.completed, snapshots=frame, input_counts=counts,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      loops=len(cbs.constraints), all_inputs_acknowledged=True,
                      fully_connected=len({tuple(sorted(a["sessions"])) for a in cbs.alignments.values()})==1
                      and len(cbs.alignments)==len(trackers))
    except BaseException as error:
        report.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-origin)
        event("failed", error=repr(error))
        raise
    finally:
        if cbs is not None:
            cbs.close()
        for node in trackers:
            node.close()
        executor.shutdown()
        for node in [*trackers, *players, *([cbs] if cbs is not None else [])]:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        write_json(args.output / "run.json", report)
        journal.close()


def render(args):
    # Reuse the paper's exact robot palette; no optimizer/graph input is read.
    sys.path.insert(0, str(ROOT / "paper/icra2027/scripts"))
    from plot_iphone_cbs_teaser import ROBOT_COLORS
    snapshots = sorted((args.input / "snapshots").glob("*.json"))
    if not snapshots:
        raise ValueError("No recorded online snapshots")
    packets = [json.loads(p.read_text()) for p in snapshots]
    elapsed = np.array([p["elapsed"] for p in packets])
    ids = sorted(packets[-1]["robots"], key=lambda s:int(s.replace("robot","")))
    palette = {r:tuple(bytes.fromhex(ROBOT_COLORS[r][1:])[::-1]) for r in ids}
    width, height, fps = 1920, 1080, 30
    video = args.output or args.input / "iphone14_online_cbsim_1080p.mp4"
    if video.exists():
        raise FileExistsError(video)
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    def text(image, label, xy, size=.65, color=(45,45,45), thickness=1):
        cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
    last_index, canvas = -1, None
    try:
        for frame in range(round(args.duration*fps)):
            # Uniform time compression; no cherry-picking or smoothing between CBS results.
            t = elapsed[0] + (elapsed[-1]-elapsed[0])*min(frame/(max(1,(args.duration-3)*fps)),1.)
            index = max(0, int(np.searchsorted(elapsed, t, side="right"))-1)
            if index != last_index:
                packet = packets[index]
                canvas = np.full((height,width,3), 250, np.uint8)
                text(canvas, f"CBSim | {len(ids)}-robot online mapping", (28,44), 1.12, thickness=2)
                text(canvas, "Fresh self-collected dataset replay | interleaved frontends + periodic CBSim", (30,76), .64)
                text(canvas, f"Run {packet['elapsed']:.1f}s  |  Completed CBSim solves {packet['cycle']}  |  Verified loops {packet['loops']}", (30,109), .65)
                operation = ("CBSim optimizing..." if packet["cbs_running"] else
                             "Frontends growing the map" if any(not v["finished"] for v in packet["robots"].values())
                             else "Final online map")
                text(canvas, operation, (1040,109), .62,
                     (20,130,180) if packet["cbs_running"] else (70,110,30))
                with np.load(snapshots[index].with_suffix(".npz")) as data:
                    groups = {}
                    for robot in ids:
                        group = tuple(packet["robots"][robot]["group"])
                        groups.setdefault(group, []).append(robot)
                    ordered_groups = sorted(groups.items(), key=lambda item:min(ids.index(r) for r in item[1]))
                    for (group,members), (left,top,cell_w,cell_h) in map_viewports(ordered_groups):
                        members = sorted(members, key=ids.index)
                        cv2.rectangle(canvas,(left,top),(left+cell_w-8,top+cell_h-8),(235,235,235),1)
                        label = "Shared map" if len(members)==len(ids) else "+".join(f"R{int(r[5:])+1:02d}" for r in members)
                        if 3 < len(members) < len(ids):
                            label = f"Aligned group | {len(members)} robots"
                        text(canvas,label,(left+10,top+24),.55)
                        paths = [data[r+"_path"][:,[0,2]] for r in members if r+"_path" in data]
                        display_maps = {r: display_cloud(data[r+"_points"])[:,[0,2]]
                                        for r in members if r+"_points" in data}
                        clouds = list(display_maps.values())
                        if not paths or not sum(len(p) for p in paths):
                            continue
                        all_paths = np.concatenate(paths)
                        low,high = all_paths.min(0),all_paths.max(0)
                        if clouds and sum(len(p) for p in clouds):
                            points = np.concatenate(clouds)
                            low = np.minimum(low, np.quantile(points,.01,axis=0))
                            high = np.maximum(high, np.quantile(points,.99,axis=0))
                        extent = np.maximum(high-low,.1)
                        scale = min((cell_w-55)/extent[0],(cell_h-75)/extent[1])*.94
                        center = (low+high)/2
                        def project(p):
                            q = (p-center)*scale
                            return np.rint(q*np.array([1,-1])+[left+cell_w/2,top+cell_h/2+10]).astype(np.int32)
                        for robot in members:
                            if robot+"_path" not in data:
                                continue
                            color = palette[robot]
                            path = project(data[robot+"_path"][:,[0,2]])
                            points = project(display_maps[robot])
                            valid = (points[:,0]>left+8)&(points[:,0]<left+cell_w-16)&(points[:,1]>top+32)&(points[:,1]<top+cell_h-16)
                            faint = tuple(int(.48*c+.52*250) for c in color)
                            for point in points[valid]:
                                cv2.circle(canvas,tuple(point),1,faint,-1)
                            cv2.polylines(canvas,[path],False,(255,255,255),5,cv2.LINE_AA)
                            cv2.polylines(canvas,[path],False,color,2,cv2.LINE_AA)
                            if len(path): cv2.circle(canvas,tuple(path[-1]),5,color,-1,cv2.LINE_AA)
                for i,robot in enumerate(ids):
                    left,top = 1440+(i%2)*235, 126+(i//2)*128
                    info = packet["robots"][robot]
                    image = cv2.imread(str(args.input/info["image"])) if info["image"] else None
                    if info["image"] and image is None:
                        raise FileNotFoundError(args.input/info["image"])
                    canvas[top+22:top+123,left:left+225] = 230
                    if image is not None:
                        factor = min(225/image.shape[1], 101/image.shape[0])
                        thumb_w, thumb_h = round(image.shape[1]*factor), round(image.shape[0]*factor)
                        x, y = left+(225-thumb_w)//2, top+22+(101-thumb_h)//2
                        canvas[y:y+thumb_h,x:x+thumb_w] = cv2.resize(image,(thumb_w,thumb_h))
                    state = "done" if info["finished"] else ("tracking" if info.get("initialized") else "init")
                    text(canvas,f"R{i+1:02d}  {info['frames']}/{info['expected_frames']}  {state}",
                         (left+3,top+16),.43,palette[robot])
                    cv2.rectangle(canvas,(left,top+22),(left+225,top+123),palette[robot],2)
                text(canvas, "Separate panels denote unaligned groups; panels merge only after a completed CBSim solve.", (28,1020),.61)
                speed = (elapsed[-1]-elapsed[0])/max(args.duration-3,1)
                text(canvas, f"Playback {speed:.1f}x | no future poses or loop closures | native XZ view | no ground truth", (28,1054),.61)
                if index in {0,len(packets)//2,len(packets)-1}:
                    cv2.imwrite(str(args.input/f"preview_{index:06d}.jpg"),canvas)
                last_index = index
            encoder.stdin.write(canvas.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait()!=0:
        raise RuntimeError("ffmpeg failed")
    metadata_path = video.with_suffix(".json") if args.output else args.input/"video.json"
    write_json(metadata_path,dict(video=str(video),width=width,height=height,fps=fps,
        duration=args.duration,source_snapshots=len(snapshots),timeline="recorded wall time",
        display_filter="Per-robot current cloud radius <= median + 10 MAD; raw snapshots unchanged; all trajectory positions retained",
        future_estimates_used=False,interpolation="none",group_frames="independent before completed CBS alignment",
        final_hold_seconds=3,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(video)


def audit(args):
    """Check causal display history and retained solver inputs, not map quality."""
    report = json.loads((args.input / "run.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Cannot certify an incomplete replay")
    events = [json.loads(line) for line in (args.input / "events.jsonl").read_text().splitlines()]
    completed = {e["cycle"]: e for e in events if e["event"] == "cbs_cycle_completed"}
    done_at = next(e["elapsed"] for e in events if e["event"] == "all_inputs_finished")
    assert len(completed) == report["completed_cbs_cycles"] >= 2
    snapshots = sorted((args.input / "snapshots").glob("*.json"))
    previous = None
    for path in snapshots:
        packet = json.loads(path.read_text())
        if previous:
            assert packet["elapsed"] >= previous["elapsed"]
            assert packet["cycle"] >= previous["cycle"]
            for robot, info in packet["robots"].items():
                assert info["frames"] >= previous["robots"][robot]["frames"]
                assert info["frames"] <= info["expected_frames"]
        for cycle in range(1, packet["cycle"]+1):
            assert completed[cycle]["elapsed"] <= packet["elapsed"]
        for info in packet["robots"].values():
            if info["image"]:
                assert (args.input/info["image"]).is_file(), info["image"]
        with np.load(path.with_suffix(".npz")) as data:
            assert all(np.isfinite(data[key]).all() for key in data.files)
        previous = packet
    assert previous["cycle"] == len(completed)
    assert all(v["frames"] == v["expected_frames"] for v in previous["robots"].values())
    for cycle, event in completed.items():
        directory = args.input / "cbs" / Path(event["output_dir"]).name
        command = json.loads((directory / "command.json").read_text())
        assert "--iterations=100" in command and "--stage_mode=alternating" in command
        assert "--pose_block_iterations=50" in command
        assert "--anchor_block_iterations=50" in command
        graph = json.loads((directory / "input_keyframes_unoptimized.json").read_text())
        assert len(graph["vertices"]) == event["poses"]
        assert (directory / "cbs.csv").is_file()
        with (directory / "cbs_belief_convergence.csv").open() as stream:
            history = list(csv.DictReader(stream))
        assert [int(row["iteration"]) for row in history] == list(range(1, 101))
        assert [row["stage"] for row in history] == ["pose"]*50 + ["anchor"]*50
    online_solves = sum(e["elapsed"] < done_at for e in completed.values())
    assert online_solves > 0
    result = dict(status="passed", snapshots=len(snapshots),
        periodic_snapshots=sum(e["event"] == "cbs_cycle_started" for e in events),
        completed_component_solves=len(completed), solves_before_inputs_finished=online_solves,
        processed_frames=sum(report["input_counts"].values()),
        fully_connected=report["fully_connected"],
        checks=["monotonic frame counts", "all source frames acknowledged",
                "finite recorded geometry", "no future CBS solution displayed",
                "every completed graph and 50/50 solver command retained; stage history verified",
                "CBS solves completed while inputs were active"],
        scope="Causal recording integrity; not an accuracy or real-time performance benchmark")
    write_json(args.input / "audit.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="mode",required=True)
    capture=commands.add_parser("run")
    capture.add_argument("--manifest",type=Path,required=True)
    capture.add_argument("--output",type=Path,required=True)
    capture.add_argument("--robot-count",type=int,default=14)
    capture.add_argument("--max-frames",type=int,default=0)
    capture.add_argument("--cbs-period",type=float,default=10.)
    capture.add_argument("--record-period",type=float,default=1.)
    capture.add_argument("--settle-seconds",type=float,default=30.)
    capture.add_argument("--timeout",type=float,default=1800.)
    capture.add_argument("--max-points",type=int,default=2500)
    movie=commands.add_parser("render")
    movie.add_argument("--input",type=Path,required=True)
    movie.add_argument("--duration",type=float,default=90.)
    movie.add_argument("--output",type=Path)
    check=commands.add_parser("audit")
    check.add_argument("--input",type=Path,required=True)
    args=parser.parse_args()
    {"run": run, "render": render, "audit": audit}[args.mode](args)


if __name__=="__main__":
    main()

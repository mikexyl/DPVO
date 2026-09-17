from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from dpvo.loop_closure.centralized import (
    CentralizedPgoResult,
    RobotMapConstraint,
    Sim3,
)
from dpvo.loop_closure.pose_graph import (
    build_keyframe_graph,
    write_g2o,
    write_json,
)
from dpvo_multi_robot_interfaces.msg import (
    InterRobotLoopClosure,
    RobotMapTransform,
)


def _sim3_from_transform(transform, scale=1.0):
    quaternion = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]
    return Sim3(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        Rotation.from_quat(quaternion).as_matrix(),
        scale,
    )


def _pose_to_sim3(pose):
    return Sim3(
        [pose.position.x, pose.position.y, pose.position.z],
        Rotation.from_quat(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        ).as_matrix(),
        1.0,
    )


def _read_summary(path: Path):
    output = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            solution = row.pop("solution")
            output[solution] = {key: float(value) for key, value in row.items()}
    return output


def _read_trajectory(path: Path):
    output = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["robot_id"],
                row["session_id"],
                int(row["keyframe_id"]),
            )
            output[key] = Sim3(
                [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                Rotation.from_quat(
                    [
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                        float(row["qw"]),
                    ]
                ).as_matrix(),
                float(row["scale"]),
            )
    return output


def _map_transforms(graph, trajectory):
    first_vertices = {}
    for vertex in graph.vertices:
        key = (vertex.robot_id, vertex.session_id)
        current = first_vertices.get(key)
        if current is None or vertex.keyframe_id < current.keyframe_id:
            first_vertices[key] = vertex
    output = {}
    for key, vertex in first_vertices.items():
        global_pose = trajectory[
            (vertex.robot_id, vertex.session_id, vertex.keyframe_id)
        ]
        output[key] = global_pose.compose(vertex.estimate.inverse())
    return output


def _sim3_dict(transform):
    return {
        "translation": transform.translation.tolist(),
        "quaternion_xyzw": Rotation.from_matrix(
            transform.rotation
        ).as_quat().tolist(),
        "scale": transform.scale,
    }


class CbsPgoNode(Node):
    def __init__(self):
        super().__init__("cbs_pgo")
        self._declare_parameters()
        self.robot_ids = list(self.get_parameter("robot_ids").value)
        self.lock = threading.RLock()
        self.constraints = []
        self.constraint_ids = set()
        self.paths = {}
        self.sessions_by_robot = {}
        self.active_sessions = {}
        self.done_robots = set()
        self.online_transforms = {}
        self.last_input_update = time.monotonic()
        self.last_signature = None
        self.worker = None
        self.process = None
        self.run_index = 0
        self.rerun_enabled = False

        self._start_rerun()

        self.constraint_subscription = self.create_subscription(
            InterRobotLoopClosure,
            self.get_parameter("constraint_topic").value,
            self._constraint,
            50,
        )
        self.central_transform_subscription = self.create_subscription(
            RobotMapTransform,
            self.get_parameter("central_transform_topic").value,
            self._central_transform,
            50,
        )
        path_suffix = self.get_parameter("path_topic_suffix").value.lstrip("/")
        done_suffix = self.get_parameter("done_topic_suffix").value.lstrip("/")
        self.path_subscriptions = []
        self.done_subscriptions = []
        self.cbs_path_publishers = {}
        self.baseline_path_publishers = {}
        self.explicit_baseline_path_publishers = {}
        for robot_id in self.robot_ids:
            self.path_subscriptions.append(
                self.create_subscription(
                    PathMessage,
                    f"/{robot_id}/{path_suffix}",
                    lambda message, robot=robot_id: self._path(robot, message),
                    10,
                )
            )
            self.done_subscriptions.append(
                self.create_subscription(
                    Bool,
                    f"/{robot_id}/{done_suffix}",
                    lambda message, robot=robot_id: self._done(robot, message),
                    10,
                )
            )
            self.cbs_path_publishers[robot_id] = self.create_publisher(
                PathMessage,
                f"/{robot_id}/dpvo/cbs_global_path",
                10,
            )
            self.baseline_path_publishers[robot_id] = self.create_publisher(
                PathMessage,
                f"/{robot_id}/dpvo/cbs_centralized_global_path",
                10,
            )
            self.explicit_baseline_path_publishers[robot_id] = self.create_publisher(
                PathMessage,
                f"/{robot_id}/dpvo/cbs_explicit_anchor_centralized_global_path",
                10,
            )

        self.transform_publisher = self.create_publisher(
            RobotMapTransform,
            self.get_parameter("transform_topic").value,
            20,
        )
        self.comparison_publisher = self.create_publisher(
            String,
            self.get_parameter("comparison_topic").value,
            10,
        )
        self.trigger_service = self.create_service(
            Trigger,
            self.get_parameter("trigger_service").value,
            self._trigger,
        )
        self.timer = self.create_timer(1.0, self._timer)

    def _declare_parameters(self):
        self.declare_parameter(
            "constraint_topic", "/dpvo_multi_robot/loop_closure"
        )
        self.declare_parameter(
            "central_transform_topic", "/dpvo_multi_robot/map_transforms"
        )
        self.declare_parameter(
            "transform_topic", "/dpvo_multi_robot/cbs/map_transforms"
        )
        self.declare_parameter("comparison_topic", "/dpvo_multi_robot/cbs/comparison")
        self.declare_parameter("trigger_service", "/dpvo_multi_robot/cbs/run")
        self.declare_parameter("robot_ids", ["robot0", "robot1", "robot2"])
        self.declare_parameter("anchor_robot_id", "robot0")
        self.declare_parameter("path_topic_suffix", "dpvo/path")
        self.declare_parameter("done_topic_suffix", "dpvo/player_done")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("cbs_executable", "")
        self.declare_parameter("iterations", 200)
        self.declare_parameter("stage_mode", "alternating")
        self.declare_parameter("anchor_start_iteration", 30)
        self.declare_parameter("anchor_stage_probability", 0.5)
        self.declare_parameter("pose_warmup_iterations", 0)
        self.declare_parameter("pose_block_iterations", 20)
        self.declare_parameter("anchor_block_iterations", 20)
        self.declare_parameter("target_hellinger", 0.1)
        self.declare_parameter("contract_alpha", 0.95)
        self.declare_parameter("d_reset", 0.1)
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("settle_seconds", 30.0)
        self.declare_parameter("run_centralized_baseline", True)
        self.declare_parameter("run_explicit_anchor_centralized_baseline", True)
        self.declare_parameter("timeout_seconds", 0.0)
        self.declare_parameter("pose_graph_odometry_weight", 100.0)
        self.declare_parameter("rerun_connect", "")
        self.declare_parameter("rerun_recording_id", "")

    def _start_rerun(self):
        connect_url = self.get_parameter("rerun_connect").value
        if not connect_url:
            return
        global rr
        import rerun as rr
        recording_id = self.get_parameter("rerun_recording_id").value or None
        rr.init("DPVO Multi Robot", recording_id=recording_id, strict=True)
        rr.connect_grpc(connect_url)
        self.rerun_enabled = True

    def _activate_session(self, robot_id, session_id):
        current = self.active_sessions.get(robot_id)
        if current == session_id:
            return True
        if current is not None and session_id < current:
            return False
        self.active_sessions[robot_id] = session_id
        self.sessions_by_robot[robot_id] = session_id
        if current is None:
            return True
        self.constraints = [
            constraint
            for constraint in self.constraints
            if robot_id
            not in (constraint.query_robot[0], constraint.match_robot[0])
        ]
        self.constraint_ids = {
            identity
            for identity in self.constraint_ids
            if robot_id not in (identity[0], identity[3])
        }
        self.online_transforms = {
            key: value
            for key, value in self.online_transforms.items()
            if key[0] != robot_id
        }
        self.get_logger().info(
            f"CBS input session changed for {robot_id}: {current} -> {session_id}"
        )
        return True

    def _constraint(self, message):
        identity = (
            message.query_robot_id,
            message.query_session_id,
            int(message.query_keyframe_id),
            message.match_robot_id,
            message.match_session_id,
            int(message.match_keyframe_id),
        )
        with self.lock:
            if not self._activate_session(
                message.query_robot_id, message.query_session_id
            ) or not self._activate_session(
                message.match_robot_id, message.match_session_id
            ):
                return
            if identity in self.constraint_ids:
                return
            self.constraint_ids.add(identity)
            weight = max(float(message.inlier_ratio), 0.05) * min(
                float(message.inliers) / 30.0, 3.0
            )
            self.constraints.append(
                RobotMapConstraint(
                    query_robot=(
                        message.query_robot_id,
                        message.query_session_id,
                    ),
                    match_robot=(
                        message.match_robot_id,
                        message.match_session_id,
                    ),
                    query_pose=Sim3.from_pose(message.query_pose),
                    match_pose=Sim3.from_pose(message.match_pose),
                    query_to_match=_sim3_from_transform(
                        message.query_to_match, message.scale
                    ),
                    weight=weight,
                    query_keyframe_id=int(message.query_keyframe_id),
                    match_keyframe_id=int(message.match_keyframe_id),
                    bow_score=float(message.bow_score),
                    inliers=int(message.inliers),
                    inlier_ratio=float(message.inlier_ratio),
                    verification_method=message.verification_method,
                )
            )
            self.last_input_update = time.monotonic()

    def _path(self, robot_id, message):
        with self.lock:
            previous = self.paths.get(robot_id)
            self.paths[robot_id] = message
            if previous is None or len(previous.poses) != len(message.poses):
                self.last_input_update = time.monotonic()

    def _done(self, robot_id, message):
        if not message.data:
            return
        with self.lock:
            if robot_id not in self.done_robots:
                self.get_logger().info(f"CBS observed completion of {robot_id}")
            self.done_robots.add(robot_id)
            self.last_input_update = time.monotonic()

    def _central_transform(self, message):
        with self.lock:
            self.online_transforms[(message.robot_id, message.session_id)] = (
                _sim3_from_transform(message.local_map_to_global, message.scale)
            )

    @staticmethod
    def _path_poses(message):
        return [_pose_to_sim3(pose.pose) for pose in message.poses]

    @staticmethod
    def _path_timestamps(message):
        return [
            pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9
            for pose in message.poses
        ]

    def _signature(self):
        return (
            len(self.constraints),
            tuple(sorted((robot, len(path.poses)) for robot, path in self.paths.items())),
            tuple(sorted(self.sessions_by_robot.items())),
        )

    def _ready(self, require_done):
        if require_done and set(self.robot_ids) != self.done_robots:
            return False, "not all robots have finished"
        if any(robot not in self.paths or not self.paths[robot].poses for robot in self.robot_ids):
            return False, "one or more robot paths are empty"
        if any(robot not in self.sessions_by_robot for robot in self.robot_ids):
            return False, "one or more robot sessions are unknown"
        if not self.constraints:
            return False, "no inter-robot constraints are available"
        return True, "ready"

    def _launch(self, force=False):
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return False, "CBS is already running"
            ready, reason = self._ready(require_done=not force)
            if not ready:
                return False, reason
            settle = float(self.get_parameter("settle_seconds").value)
            if not force and time.monotonic() - self.last_input_update < settle:
                return False, "waiting for the loop-verification queue to settle"
            signature = self._signature()
            if not force and signature == self.last_signature:
                return False, "this graph snapshot was already optimized"
            constraints = list(self.constraints)
            paths = {
                robot: self._path_poses(message)
                for robot, message in self.paths.items()
            }
            sessions = dict(self.sessions_by_robot)
            online = dict(self.online_transforms)
            timestamps = {robot: self._path_timestamps(message)
                          for robot, message in self.paths.items()}
            self.last_signature = signature
            self.worker = threading.Thread(
                target=self._run,
                args=(constraints, paths, sessions, online, timestamps),
                daemon=True,
            )
            self.worker.start()
        return True, "CBS optimization scheduled"

    def _timer(self):
        launched, reason = self._launch(force=False)
        if launched:
            self.get_logger().info(reason)

    def _trigger(self, _request, response):
        response.success, response.message = self._launch(force=True)
        return response

    def _command(self, graph_path, output_dir):
        configured = self.get_parameter("cbs_executable").value
        resolved = shutil.which(configured)
        executable = Path(resolved or configured).expanduser()
        if not executable.is_file():
            raise FileNotFoundError(f"CBS executable does not exist: {executable}")
        parameters = {
            "input_graph": graph_path,
            "output_dir": output_dir,
            "iterations": int(self.get_parameter("iterations").value),
            "stage_mode": self.get_parameter("stage_mode").value,
            "anchor_start_iteration": int(
                self.get_parameter("anchor_start_iteration").value
            ),
            "anchor_stage_probability": float(
                self.get_parameter("anchor_stage_probability").value
            ),
            "pose_warmup_iterations": int(
                self.get_parameter("pose_warmup_iterations").value
            ),
            "pose_block_iterations": int(
                self.get_parameter("pose_block_iterations").value
            ),
            "anchor_block_iterations": int(
                self.get_parameter("anchor_block_iterations").value
            ),
            "target_hellinger": float(
                self.get_parameter("target_hellinger").value
            ),
            "contract_alpha": float(self.get_parameter("contract_alpha").value),
            "d_reset": float(self.get_parameter("d_reset").value),
            "random_seed": int(self.get_parameter("random_seed").value),
            "run_centralized": str(
                bool(self.get_parameter("run_centralized_baseline").value)
            ).lower(),
            "run_explicit_anchor_centralized": str(
                bool(
                    self.get_parameter(
                        "run_explicit_anchor_centralized_baseline"
                    ).value
                )
            ).lower(),
            "write_rerun_rrd": "false",
            "rerun_stream": "false",
        }
        return [str(executable)] + [
            f"--{key}={value}" for key, value in parameters.items()
        ]

    def _run(self, constraints, paths, sessions, online, timestamps=None):
        started = time.monotonic()
        configured_output = self.get_parameter("output_dir").value
        try:
            if not configured_output:
                raise ValueError("CBS output_dir is required")
            output_dir = Path(configured_output).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            graph = build_keyframe_graph(
                constraints,
                CentralizedPgoResult({}, True, 0.0, 0.0),
                paths,
                sessions,
                self.get_parameter("anchor_robot_id").value,
                odometry_weight=float(
                    self.get_parameter("pose_graph_odometry_weight").value
                ),
                align_to_global=False,
                timestamps=timestamps if timestamps is not None else {
                    robot: self._path_timestamps(message)
                    for robot, message in self.paths.items()
                },
            )
            if graph.metadata["skipped_inter_robot_loops"]:
                raise RuntimeError(
                    "CBS input graph skipped %d inter-robot constraints"
                    % graph.metadata["skipped_inter_robot_loops"]
                )
            graph_path = output_dir / "input_keyframes_unoptimized.json"
            write_json(graph, graph_path)
            write_g2o(graph, graph_path.with_suffix(".g2o"))
            command = self._command(graph_path, output_dir)
            self.get_logger().info(
                "Starting CBS on %d vertices / %d edges / %d loops"
                % (len(graph.vertices), len(graph.edges), len(constraints))
            )
            with self.lock:
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                process = self.process
            timeout = float(self.get_parameter("timeout_seconds").value)
            stdout, _ = process.communicate(timeout=timeout if timeout > 0 else None)
            (output_dir / "cbs.log").write_text(stdout)
            if process.returncode != 0:
                raise RuntimeError(
                    f"CBS exited with code {process.returncode}; see cbs.log"
                )
            duration = time.monotonic() - started
            self._publish_results(graph, output_dir, online, duration)
            self.get_logger().info(
                f"CBS comparison complete in {duration:.3f}s: {output_dir}"
            )
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
            self.get_logger().error("CBS optimization timed out")
        except Exception as error:
            self.get_logger().error(f"CBS optimization failed: {error}")
        finally:
            with self.lock:
                self.process = None

    def _publish_results(self, graph, output_dir, online, duration):
        cbs_name = "cbs"
        summary = _read_summary(output_dir / "summary.csv")
        cbs_trajectory = _read_trajectory(output_dir / "cbs.csv")
        baseline_path = output_dir / "centralized.csv"
        baseline_trajectory = (
            _read_trajectory(baseline_path) if baseline_path.is_file() else None
        )
        explicit_baseline_path = output_dir / "centralized_explicit_anchors.csv"
        explicit_baseline_trajectory = (
            _read_trajectory(explicit_baseline_path)
            if explicit_baseline_path.is_file()
            else None
        )
        cbs_transforms = _map_transforms(graph, cbs_trajectory)
        baseline_transforms = (
            _map_transforms(graph, baseline_trajectory)
            if baseline_trajectory is not None
            else {}
        )
        explicit_baseline_transforms = (
            _map_transforms(graph, explicit_baseline_trajectory)
            if explicit_baseline_trajectory is not None
            else {}
        )
        cbs_metrics = summary[cbs_name]
        constraint_count = int(graph.metadata["inter_robot_loop_count"])
        residual_norm = math.sqrt(max(2.0 * cbs_metrics["graph_error"], 0.0))
        for key, transform in cbs_transforms.items():
            self._publish_transform(
                key,
                transform,
                constraint_count,
                cbs_metrics["graph_error"],
                residual_norm,
            )
        self._publish_paths(cbs_trajectory, self.cbs_path_publishers)
        if baseline_trajectory is not None:
            self._publish_paths(
                baseline_trajectory, self.baseline_path_publishers
            )
        if explicit_baseline_trajectory is not None:
            self._publish_paths(
                explicit_baseline_trajectory,
                self.explicit_baseline_path_publishers,
            )

        comparison = {
            "runtime_seconds": duration,
            "vertex_count": len(graph.vertices),
            "edge_count": len(graph.edges),
            "constraint_count": constraint_count,
            "summary": summary,
            "map_transforms": {},
            "note": (
                "CBS and cbs_full_graph_centralized use the same keyframe graph "
                "objective. cbs_full_graph_explicit_anchor_centralized uses the "
                "same measurements with separate local-pose and robot-anchor "
                "variables. online_map_centralized is the existing map-level PGO; "
                "its transform deltas are comparable, but its cost is not."
            ),
        }
        for key, cbs_transform in cbs_transforms.items():
            label = f"{key[0]}:{key[1]}"
            entry = {"cbs": _sim3_dict(cbs_transform)}
            if key in baseline_transforms:
                entry["cbs_full_graph_centralized"] = _sim3_dict(
                    baseline_transforms[key]
                )
            if key in explicit_baseline_transforms:
                entry["cbs_full_graph_explicit_anchor_centralized"] = _sim3_dict(
                    explicit_baseline_transforms[key]
                )
            central = online.get(key)
            if central is not None:
                delta = central.inverse().compose(cbs_transform)
                entry["online_map_centralized"] = _sim3_dict(central)
                entry["cbs_minus_online"] = {
                    "translation_norm": float(np.linalg.norm(delta.translation)),
                    "rotation_radians": float(
                        Rotation.from_matrix(delta.rotation).magnitude()
                    ),
                    "log_scale": float(np.log(delta.scale)),
                    "scale_ratio": delta.scale,
                }
            comparison["map_transforms"][label] = entry

        comparison_path = output_dir / "comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
        self.comparison_publisher.publish(
            String(data=json.dumps(comparison, separators=(",", ":")))
        )
        self._log_rerun(
            cbs_trajectory,
            baseline_trajectory,
            explicit_baseline_trajectory,
            comparison,
            cbs_transforms,
            online,
        )

    def _result_frame(self):
        return self.get_parameter("global_frame").value

    def _publish_transform(self, key, transform, count, cost, residual_norm):
        message = RobotMapTransform()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._result_frame()
        message.robot_id, message.session_id = key
        message.local_map_to_global.translation.x = float(transform.translation[0])
        message.local_map_to_global.translation.y = float(transform.translation[1])
        message.local_map_to_global.translation.z = float(transform.translation[2])
        quaternion = Rotation.from_matrix(transform.rotation).as_quat()
        message.local_map_to_global.rotation.x = float(quaternion[0])
        message.local_map_to_global.rotation.y = float(quaternion[1])
        message.local_map_to_global.rotation.z = float(quaternion[2])
        message.local_map_to_global.rotation.w = float(quaternion[3])
        message.scale = transform.scale
        message.constraint_count = count
        message.cost = cost
        message.residual_norm = residual_norm
        self.transform_publisher.publish(message)

    def _publish_paths(self, trajectory, publishers):
        stamp = self.get_clock().now().to_msg()
        frame = self.get_parameter("global_frame").value
        for robot_id in self.robot_ids:
            rows = sorted(
                (
                    (key, transform)
                    for key, transform in trajectory.items()
                    if key[0] == robot_id
                ),
                key=lambda item: (item[0][1], item[0][2]),
            )
            message = PathMessage()
            message.header.stamp = stamp
            message.header.frame_id = frame
            for _key, transform in rows:
                pose = PoseStamped()
                pose.header = message.header
                pose.pose.position.x = float(transform.translation[0])
                pose.pose.position.y = float(transform.translation[1])
                pose.pose.position.z = float(transform.translation[2])
                quaternion = Rotation.from_matrix(transform.rotation).as_quat()
                pose.pose.orientation.x = float(quaternion[0])
                pose.pose.orientation.y = float(quaternion[1])
                pose.pose.orientation.z = float(quaternion[2])
                pose.pose.orientation.w = float(quaternion[3])
                message.poses.append(pose)
            publishers[robot_id].publish(message)

    @staticmethod
    def _rerun_color(robot_id):
        return {
            "robot0": [0, 170, 255],
            "robot1": [255, 95, 85],
            "robot2": [110, 220, 120],
        }.get(robot_id, [220, 180, 70])

    def _log_rerun(
        self,
        cbs_trajectory,
        baseline_trajectory,
        explicit_baseline_trajectory,
        comparison,
        cbs_transforms,
        online,
    ):
        if not self.rerun_enabled:
            return
        try:
            self.run_index += 1
            rr.set_time("cbs_run", sequence=self.run_index)
            for solution, trajectory in (
                ("cbs", cbs_trajectory),
                ("full_graph_centralized", baseline_trajectory),
                (
                    "full_graph_explicit_anchor_centralized",
                    explicit_baseline_trajectory,
                ),
            ):
                if trajectory is None:
                    continue
                for robot_id in self.robot_ids:
                    points = np.asarray(
                        [
                            transform.translation
                            for key, transform in sorted(trajectory.items())
                            if key[0] == robot_id
                        ]
                    )
                    if points.size:
                        rr.log(
                            f"world/comparison/{solution}/{robot_id}",
                            rr.LineStrips3D(
                                [points],
                                colors=self._rerun_color(robot_id),
                                radii=rr.Radius.ui_points(3.0),
                            ),
                        )
            for name, metrics in comparison["summary"].items():
                for metric in (
                    "graph_error",
                    "inter_translation_rmse",
                    "inter_log_scale_rmse",
                ):
                    rr.log(
                        f"metrics/cbs_comparison/{name}/{metric}",
                        rr.Scalars([metrics[metric]]),
                    )
            for key, transform in cbs_transforms.items():
                rr.log(
                    f"metrics/cbs_comparison/map_scale/{key[0]}/cbs",
                    rr.Scalars([transform.scale]),
                )
                if key in online:
                    rr.log(
                        f"metrics/cbs_comparison/map_scale/{key[0]}/online",
                        rr.Scalars([online[key].scale]),
                    )
        except Exception as error:
            self.get_logger().warning(f"CBS Rerun logging failed: {error}")

    def close(self):
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=10)
        if self.rerun_enabled:
            rr.disconnect()
            self.rerun_enabled = False


def main(args=None):
    rclpy.init(args=args)
    node = CbsPgoNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from dpvo.loop_closure.centralized import (
    CentralizedPgoResult,
    CentralizedRobotMapPGO,
    RobotMapConstraint,
    Sim3,
)
from dpvo.loop_closure.pose_graph import (
    build_keyframe_graph,
    build_map_graph,
    split_keyframe_graph_by_robot,
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


class CentralizedPgoNode(Node):
    def __init__(self):
        super().__init__("centralized_pgo")
        self.declare_parameter(
            "constraint_topic",
            "/dpvo_multi_robot/loop_closure",
        )
        self.declare_parameter(
            "transform_topic",
            "/dpvo_multi_robot/map_transforms",
        )
        self.declare_parameter("robot_ids", ["robot0", "robot1", "robot2"])
        self.declare_parameter("anchor_robot_id", "robot0")
        self.declare_parameter("path_topic_suffix", "dpvo/path")
        self.declare_parameter("global_path_topic_suffix", "dpvo/global_path")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("output_path", "")
        self.declare_parameter("pose_graph_output", "")
        self.declare_parameter("pose_graph_export_period", 0.0)
        self.declare_parameter("pose_graph_odometry_weight", 100.0)
        self.declare_parameter("rerun_connect", "")
        self.declare_parameter("rerun_recording_id", "")

        self.lock = threading.RLock()
        self.constraints = []
        self.constraint_ids = set()
        self.transforms = {}
        self.paths = {}
        self.path_publishers = {}
        self.path_subscriptions = []
        self.sessions_by_robot = {}
        self.active_sessions = {}
        self.rerun_enabled = False
        self.rerun_update = 0
        self.rerun_frame = 0
        self.last_result = CentralizedPgoResult({}, True, 0.0, 0.0)
        self.last_pose_graph_signature = None

        self._start_rerun()

        self.transform_publisher = self.create_publisher(
            RobotMapTransform,
            self.get_parameter("transform_topic").value,
            20,
        )
        self.constraint_subscription = self.create_subscription(
            InterRobotLoopClosure,
            self.get_parameter("constraint_topic").value,
            self._constraint,
            50,
        )

        path_suffix = self.get_parameter("path_topic_suffix").value.lstrip("/")
        global_suffix = self.get_parameter(
            "global_path_topic_suffix"
        ).value.lstrip("/")
        for robot_id in self.get_parameter("robot_ids").value:
            self.path_subscriptions.append(
                self.create_subscription(
                    PathMessage,
                    f"/{robot_id}/{path_suffix}",
                    lambda message, robot=robot_id: self._path(robot, message),
                    10,
                )
            )
            self.path_publishers[robot_id] = self.create_publisher(
                PathMessage,
                f"/{robot_id}/{global_suffix}",
                10,
            )

        self.pose_graph_base = self._pose_graph_base()
        self.pose_graph_timer = None
        if self.pose_graph_base is not None:
            period = float(self.get_parameter("pose_graph_export_period").value)
            if period > 0.0:
                self.pose_graph_timer = self.create_timer(
                    max(period, 0.1), self.save_pose_graph
                )
                schedule = f"every {max(period, 0.1):g} seconds and at shutdown"
            else:
                schedule = "at shutdown"
            self.get_logger().info(
                "Pose-graph export enabled %s: %s_{map,keyframes}"
                % (schedule, self.pose_graph_base)
            )

    def _pose_graph_base(self):
        explicit = self.get_parameter("pose_graph_output").value
        if explicit:
            path = Path(explicit).expanduser()
            return path.with_suffix("") if path.suffix else path
        output_path = self.get_parameter("output_path").value
        if not output_path:
            return None
        path = Path(output_path).expanduser()
        path = path.with_suffix("") if path.suffix else path
        return path.with_name(f"{path.name}_pose_graph")

    def _start_rerun(self):
        connect_url = self.get_parameter("rerun_connect").value
        if not connect_url:
            return
        recording_id = self.get_parameter("rerun_recording_id").value or None
        robot_ids = self.get_parameter("robot_ids").value
        camera_views = [
            rrb.Spatial2DView(
                name=f"{robot_id} camera",
                origin=f"world/robots/{robot_id}/map/camera/image",
            )
            for robot_id in robot_ids
        ]
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Centralized multi-robot map", origin="world"),
                rrb.Tabs(*camera_views, name="Robot cameras"),
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
        rr.init(
            "DPVO Multi Robot",
            recording_id=recording_id,
            default_blueprint=blueprint,
            strict=True,
        )
        rr.connect_grpc(connect_url)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        self.rerun_enabled = True
        self.get_logger().info(
            "Rerun centralized stream: recording=%s url=%s"
            % (recording_id or "auto", connect_url)
        )

    @staticmethod
    def _pose(message):
        return Sim3.from_pose(message)

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
            if robot_id not in (
                constraint.query_robot[0],
                constraint.match_robot[0],
            )
        ]
        self.constraint_ids = {
            identity
            for identity in self.constraint_ids
            if robot_id not in (identity[0], identity[3])
        }
        self.transforms = {
            robot: transform
            for robot, transform in self.transforms.items()
            if robot[0] != robot_id
        }
        self.get_logger().info(
            f"Robot {robot_id} session changed: {current} -> {session_id}"
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
                message.query_robot_id,
                message.query_session_id,
            ) or not self._activate_session(
                message.match_robot_id,
                message.match_session_id,
            ):
                return
            if identity in self.constraint_ids:
                return
            self.constraint_ids.add(identity)
            query_robot = (message.query_robot_id, message.query_session_id)
            match_robot = (message.match_robot_id, message.match_session_id)
            weight = max(float(message.inlier_ratio), 0.05) * min(
                float(message.inliers) / 30.0,
                3.0,
            )
            self.constraints.append(
                RobotMapConstraint(
                    query_robot=query_robot,
                    match_robot=match_robot,
                    query_pose=self._pose(message.query_pose),
                    match_pose=self._pose(message.match_pose),
                    query_to_match=_sim3_from_transform(
                        message.query_to_match,
                        message.scale,
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
            anchor_id = self.get_parameter("anchor_robot_id").value
            all_robots = {
                robot
                for constraint in self.constraints
                for robot in (constraint.query_robot, constraint.match_robot)
            }
            anchor = next(
                (robot for robot in all_robots if robot[0] == anchor_id),
                None,
            )
            result = CentralizedRobotMapPGO(anchor=anchor).solve(self.constraints)
            self.last_result = result
            self.transforms = result.transforms
            self._publish_transforms(result)
            self._log_rerun_result(result)
            self._publish_all_paths()
            self._save(result)
            self.get_logger().info(
                "Central PGO: %d constraints, %d robots, residual %.6f"
                % (
                    len(self.constraints),
                    len(result.transforms),
                    result.residual_norm,
                )
            )

    def _publish_transforms(self, result):
        stamp = self.get_clock().now().to_msg()
        global_frame = self.get_parameter("global_frame").value
        for (robot_id, session_id), transform in result.transforms.items():
            message = RobotMapTransform()
            message.header.stamp = stamp
            message.header.frame_id = global_frame
            message.robot_id = robot_id
            message.session_id = session_id
            message.local_map_to_global.translation.x = float(transform.translation[0])
            message.local_map_to_global.translation.y = float(transform.translation[1])
            message.local_map_to_global.translation.z = float(transform.translation[2])
            quaternion = Rotation.from_matrix(transform.rotation).as_quat()
            message.local_map_to_global.rotation.x = float(quaternion[0])
            message.local_map_to_global.rotation.y = float(quaternion[1])
            message.local_map_to_global.rotation.z = float(quaternion[2])
            message.local_map_to_global.rotation.w = float(quaternion[3])
            message.scale = transform.scale
            message.constraint_count = len(self.constraints)
            message.cost = result.cost
            message.residual_norm = result.residual_norm
            self.transform_publisher.publish(message)

    @staticmethod
    def _rerun_color(robot_id):
        palette = {
            "robot0": [0, 170, 255],
            "robot1": [255, 95, 85],
            "robot2": [110, 220, 120],
        }
        return palette.get(robot_id, [220, 180, 70])

    def _set_rerun_time(self):
        self.rerun_update += 1
        path_frame = max(
            (len(path.poses) for path in self.paths.values()),
            default=self.rerun_frame,
        )
        self.rerun_frame = max(self.rerun_frame, path_frame)
        rr.set_time("frame", sequence=self.rerun_frame)
        rr.set_time("pgo_update", sequence=self.rerun_update)

    def _log_rerun_result(self, result):
        if not self.rerun_enabled:
            return
        self._set_rerun_time()
        for (robot_id, _session_id), transform in result.transforms.items():
            rr.log(
                f"world/robots/{robot_id}",
                rr.Transform3D(
                    translation=transform.translation,
                    mat3x3=transform.scale * transform.rotation,
                ),
            )
            rr.log(
                f"metrics/centralized_pgo/map_scale/{robot_id}",
                rr.Scalars([transform.scale]),
            )

        strips = []
        labels = []
        for constraint in self.constraints:
            query_transform = result.transforms.get(constraint.query_robot)
            match_transform = result.transforms.get(constraint.match_robot)
            if query_transform is None or match_transform is None:
                continue
            strips.append(
                np.stack(
                    [
                        query_transform.apply(constraint.query_pose.translation),
                        match_transform.apply(constraint.match_pose.translation),
                    ]
                )
            )
            labels.append(
                f"{constraint.query_robot[0]} -> {constraint.match_robot[0]}"
            )
        if strips:
            rr.log(
                "world/centralized/loop_constraints",
                rr.LineStrips3D(
                    strips,
                    colors=[[255, 190, 50]] * len(strips),
                    labels=labels,
                    radii=rr.Radius.ui_points(2.5),
                ),
            )
        rr.log(
            "metrics/centralized_pgo/constraint_count",
            rr.Scalars([len(self.constraints)]),
        )
        rr.log("metrics/centralized_pgo/cost", rr.Scalars([result.cost]))
        rr.log(
            "metrics/centralized_pgo/residual_norm",
            rr.Scalars([result.residual_norm]),
        )

    def _path(self, robot_id, message):
        with self.lock:
            self.paths[robot_id] = message
            self._publish_path(robot_id)

    def _publish_all_paths(self):
        for robot_id in self.paths:
            self._publish_path(robot_id)

    def _publish_path(self, robot_id):
        session_id = self.sessions_by_robot.get(robot_id)
        transform = self.transforms.get((robot_id, session_id))
        if transform is None:
            return
        source = self.paths[robot_id]
        output = PathMessage()
        output.header.stamp = source.header.stamp
        output.header.frame_id = self.get_parameter("global_frame").value
        alignment_rotation = Rotation.from_matrix(transform.rotation)
        for source_pose in source.poses:
            pose = PoseStamped()
            pose.header.stamp = source_pose.header.stamp
            pose.header.frame_id = output.header.frame_id
            local_position = np.array(
                [
                    source_pose.pose.position.x,
                    source_pose.pose.position.y,
                    source_pose.pose.position.z,
                ]
            )
            global_position = transform.apply(local_position)
            pose.pose.position.x = float(global_position[0])
            pose.pose.position.y = float(global_position[1])
            pose.pose.position.z = float(global_position[2])
            local_quaternion = [
                source_pose.pose.orientation.x,
                source_pose.pose.orientation.y,
                source_pose.pose.orientation.z,
                source_pose.pose.orientation.w,
            ]
            quaternion = (alignment_rotation * Rotation.from_quat(local_quaternion)).as_quat()
            pose.pose.orientation.x = float(quaternion[0])
            pose.pose.orientation.y = float(quaternion[1])
            pose.pose.orientation.z = float(quaternion[2])
            pose.pose.orientation.w = float(quaternion[3])
            output.poses.append(pose)
        self.path_publishers[robot_id].publish(output)
        if self.rerun_enabled and output.poses:
            points = np.array(
                [
                    [
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                    ]
                    for pose in output.poses
                ]
            )
            rr.log(
                f"world/centralized/trajectories/{robot_id}",
                rr.LineStrips3D(
                    [points],
                    colors=self._rerun_color(robot_id),
                    radii=rr.Radius.ui_points(3.0),
                ),
            )

    def close_rerun(self):
        if self.rerun_enabled:
            rr.disconnect()
            self.rerun_enabled = False

    @staticmethod
    def _path_poses(path):
        return [
            Sim3(
                [
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                ],
                Rotation.from_quat(
                    [
                        pose.pose.orientation.x,
                        pose.pose.orientation.y,
                        pose.pose.orientation.z,
                        pose.pose.orientation.w,
                    ]
                ).as_matrix(),
                1.0,
            )
            for pose in path.poses
        ]

    @staticmethod
    def _path_timestamps(path):
        return [
            pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9
            for pose in path.poses
        ]

    def save_pose_graph(self, force=False):
        if self.pose_graph_base is None:
            return
        with self.lock:
            signature = (
                len(self.constraints),
                tuple(sorted((robot, len(path.poses)) for robot, path in self.paths.items())),
                tuple(sorted(self.sessions_by_robot.items())),
            )
            if not force and signature == self.last_pose_graph_signature:
                return
            try:
                anchor_id = self.get_parameter("anchor_robot_id").value
                anchor = next(
                    (
                        robot
                        for robot in self.last_result.transforms
                        if robot[0] == anchor_id
                    ),
                    None,
                )
                map_graph = build_map_graph(
                    self.constraints,
                    self.last_result,
                    anchor=anchor,
                )
                keyframe_graph = build_keyframe_graph(
                    self.constraints,
                    self.last_result,
                    {
                        robot: self._path_poses(path)
                        for robot, path in self.paths.items()
                    },
                    self.sessions_by_robot,
                    anchor_id,
                    odometry_weight=float(
                        self.get_parameter("pose_graph_odometry_weight").value
                    ),
                    timestamps={
                        robot: self._path_timestamps(path)
                        for robot, path in self.paths.items()
                    },
                )
                unoptimized_graph = build_keyframe_graph(
                    self.constraints,
                    self.last_result,
                    {
                        robot: self._path_poses(path)
                        for robot, path in self.paths.items()
                    },
                    self.sessions_by_robot,
                    anchor_id,
                    odometry_weight=float(
                        self.get_parameter("pose_graph_odometry_weight").value
                    ),
                    align_to_global=False,
                    timestamps={
                        robot: self._path_timestamps(path)
                        for robot, path in self.paths.items()
                    },
                )
                map_json = Path(f"{self.pose_graph_base}_map.json")
                keyframe_json = Path(f"{self.pose_graph_base}_keyframes.json")
                unoptimized_json = Path(
                    f"{self.pose_graph_base}_keyframes_unoptimized.json"
                )
                write_json(map_graph, map_json)
                write_g2o(map_graph, map_json.with_suffix(".g2o"))
                write_json(keyframe_graph, keyframe_json)
                write_g2o(keyframe_graph, keyframe_json.with_suffix(".g2o"))
                write_json(unoptimized_graph, unoptimized_json)
                write_g2o(
                    unoptimized_graph,
                    unoptimized_json.with_suffix(".g2o"),
                )
                for robot_id, robot_graph in split_keyframe_graph_by_robot(
                    unoptimized_graph
                ).items():
                    robot_json = Path(f"{self.pose_graph_base}_{robot_id}.json")
                    write_json(robot_graph, robot_json)
                    write_g2o(robot_graph, robot_json.with_suffix(".g2o"))
                self.last_pose_graph_signature = signature
                if rclpy.ok():
                    self.get_logger().info(
                        "Pose graphs: map %dV/%dE, keyframes %dV/%dE"
                        % (
                            len(map_graph.vertices),
                            len(map_graph.edges),
                            len(keyframe_graph.vertices),
                            len(keyframe_graph.edges),
                        )
                    )
            except Exception as error:
                if rclpy.ok():
                    self.get_logger().error(f"Pose-graph export failed: {error}")

    def _save(self, result):
        output_path = self.get_parameter("output_path").value
        if not output_path:
            return
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "constraint_count": len(self.constraints),
            "cost": result.cost,
            "residual_norm": result.residual_norm,
            "constraints": [
                {
                    "query_robot": constraint.query_robot[0],
                    "query_session": constraint.query_robot[1],
                    "match_robot": constraint.match_robot[0],
                    "match_session": constraint.match_robot[1],
                    "measurement_scale": constraint.query_to_match.scale,
                    "weight": constraint.weight,
                    "query_keyframe_id": constraint.query_keyframe_id,
                    "match_keyframe_id": constraint.match_keyframe_id,
                    "bow_score": constraint.bow_score,
                    "inliers": constraint.inliers,
                    "inlier_ratio": constraint.inlier_ratio,
                    "verification_method": constraint.verification_method,
                }
                for constraint in self.constraints
            ],
            "robots": {
                f"{robot_id}:{session_id}": {
                    "translation": transform.translation.tolist(),
                    "quaternion_xyzw": Rotation.from_matrix(
                        transform.rotation
                    ).as_quat().tolist(),
                    "scale": transform.scale,
                }
                for (robot_id, session_id), transform in result.transforms.items()
            },
        }
        path.write_text(json.dumps(data, indent=2) + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = CentralizedPgoNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save_pose_graph(force=True)
        node.close_rerun()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

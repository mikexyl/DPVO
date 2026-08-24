from __future__ import annotations

from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
import torch

from dpvo.config import cfg as default_cfg
from dpvo.dpvo import DPVO
from dpvo.lietorch import SE3
from dpvo.loop_closure.distributed import DistributedLongTermLoopClosure
from dpvo.map_gauge import transform_poses_xyzw

from .transport import Ros2DistributedTransport


class MultiRobotDpvoNode(Node):
    def __init__(self):
        super().__init__("dpvo_multi_robot")
        self._declare_parameters()

        self.robot_id = self.get_parameter("robot_id").value
        requested_session = self.get_parameter("session_id").value
        self.session_id = requested_session or (
            f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        )
        self.processing_lock = threading.RLock()
        self.callback_group = ReentrantCallbackGroup()
        self.camera = None
        self.slam = None
        self.last_published_keyframe = -1
        self.path = PathMessage()
        self.path.header.frame_id = self.get_parameter("map_frame").value

        self.cfg = default_cfg.clone()
        self.cfg.merge_from_file(self.get_parameter("config").value)
        self.cfg.CLASSIC_LOOP_CLOSURE = True
        self.cfg.CLASSIC_PGO_USE_THREADS = True
        self.cfg.MAX_EDGE_AGE = self.get_parameter("max_edge_age").value
        self.cfg.LOOP_CLOSURE = self.get_parameter(
            "enable_dpvo_loop_closure"
        ).value
        self.cfg.ORB_VOCAB_PATH = self.get_parameter("orb_vocab").value
        self.cfg.LOOP_RETR_THRESH = self.get_parameter("bow_threshold").value
        self.cfg.MULTI_ROBOT_BOW_REPETITIONS = self.get_parameter(
            "bow_repetitions"
        ).value
        self.cfg.MULTI_ROBOT_BOW_NMS = self.get_parameter(
            "bow_nms_radius"
        ).value
        self.cfg.MULTI_ROBOT_TEASER_NOISE_BOUND = self.get_parameter(
            "teaser_noise_bound"
        ).value
        self.cfg.MULTI_ROBOT_TEASER_REQUIRED = self.get_parameter(
            "teaser_required"
        ).value
        self.cfg.MULTI_ROBOT_MIN_INLIERS = self.get_parameter("min_inliers").value
        self.cfg.MULTI_ROBOT_MIN_INLIER_RATIO = self.get_parameter(
            "min_inlier_ratio"
        ).value

        self.transport = Ros2DistributedTransport(
            self,
            self.robot_id,
            self.processing_lock,
            bow_topic=self.get_parameter("bow_topic").value,
            constraint_topic=self.get_parameter("constraint_topic").value,
            service_prefix=self.get_parameter("service_prefix").value,
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_parameter("pose_topic").value,
            20,
        )
        self.path_publisher = self.create_publisher(
            PathMessage,
            self.get_parameter("path_topic").value,
            5,
        )
        ack_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.frame_ack_publisher = self.create_publisher(
            Header,
            self.get_parameter("frame_ack_topic").value,
            ack_qos,
        )
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.camera_subscription = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info,
            info_qos,
            callback_group=self.callback_group,
        )
        self.image_subscription = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self._image,
            image_qos,
            callback_group=self.callback_group,
        )
        self.distributed_timer = self.create_timer(
            0.05,
            self._process_distributed,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"multi-robot DPVO ready: robot={self.robot_id}, session={self.session_id}"
        )

    def _declare_parameters(self):
        self.declare_parameter("robot_id", "robot0")
        self.declare_parameter("session_id", "")
        self.declare_parameter("network", "dpvo.pth")
        self.declare_parameter("config", "config/fast.yaml")
        self.declare_parameter("orb_vocab", "ORBvoc.txt")
        self.declare_parameter("vocabulary_id", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("image_scale", 1.0)
        self.declare_parameter("viewer", "none")
        self.declare_parameter("rerun_save", "")
        self.declare_parameter("rerun_connect", "")
        self.declare_parameter("rerun_recording_id", "")
        self.declare_parameter("rerun_entity_prefix", "")
        self.declare_parameter("enable_dpvo_loop_closure", True)
        self.declare_parameter("max_edge_age", 48)
        self.declare_parameter("bow_threshold", 0.04)
        self.declare_parameter("bow_repetitions", 3)
        self.declare_parameter("bow_nms_radius", 50)
        self.declare_parameter("teaser_noise_bound", 0.10)
        self.declare_parameter("teaser_required", False)
        self.declare_parameter("min_inliers", 30)
        self.declare_parameter("min_inlier_ratio", 0.20)
        self.declare_parameter("bow_topic", "/dpvo_multi_robot/bow")
        self.declare_parameter(
            "constraint_topic", "/dpvo_multi_robot/loop_closure"
        )
        self.declare_parameter("service_prefix", "/dpvo_multi_robot")
        self.declare_parameter("pose_topic", "dpvo/pose")
        self.declare_parameter("path_topic", "dpvo/path")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("map_frame", "map")

    def _camera_info(self, message):
        camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(message.d, dtype=np.float64)
        self.camera = (camera_matrix, distortion)

    def _process_distributed(self):
        if self.slam is None or not self.processing_lock.acquire(blocking=False):
            return
        try:
            self.slam.long_term_lc.process_transport()
        finally:
            self.processing_lock.release()

    @staticmethod
    def _decode_image(message):
        data = np.frombuffer(message.data, dtype=np.uint8)
        if message.encoding in ("mono8", "8UC1"):
            image = data.reshape(message.height, message.step)[:, : message.width]
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if message.encoding in ("bgr8", "rgb8"):
            image = data.reshape(message.height, message.step)[:, : message.width * 3]
            image = image.reshape(message.height, message.width, 3)
            if message.encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        raise ValueError(f"unsupported image encoding: {message.encoding}")

    @torch.no_grad()
    def _image(self, message):
        if self.camera is None:
            self._publish_ack(message)
            return
        with self.processing_lock:
            camera_matrix, distortion = self.camera
            image = self._decode_image(message)
            if distortion.size and np.any(distortion):
                image = cv2.undistort(image, camera_matrix, distortion)

            scale = self.get_parameter("image_scale").value
            if not 0.0 < scale <= 1.0:
                raise ValueError("image_scale must be in (0, 1]")
            if scale != 1.0:
                image = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            height, width = image.shape[:2]
            image = image[: height - height % 16, : width - width % 16]
            intrinsics_np = np.array(
                [
                    camera_matrix[0, 0],
                    camera_matrix[1, 1],
                    camera_matrix[0, 2],
                    camera_matrix[1, 2],
                ],
                dtype=np.float32,
            ) * scale
            image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics_np).cuda()

            if self.slam is None:
                self.slam = self._create_slam(image_tensor.shape[1:])

            timestamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.slam(timestamp, image_tensor, intrinsics)
            self._publish_pose(message)
            self._publish_ack(message)

    def _publish_ack(self, image_message):
        acknowledgement = Header()
        acknowledgement.stamp = image_message.header.stamp
        acknowledgement.frame_id = self.robot_id
        self.frame_ack_publisher.publish(acknowledgement)

    def _create_slam(self, image_shape):
        vocabulary_id = self.get_parameter("vocabulary_id").value or None

        def loop_closure_factory(cfg, patchgraph):
            return DistributedLongTermLoopClosure(
                cfg,
                patchgraph,
                transport=self.transport,
                robot_id=self.robot_id,
                session_id=self.session_id,
                vocabulary_id=vocabulary_id,
            )

        viewer = self.get_parameter("viewer").value
        if viewer == "none":
            viewer = None
        rerun_save = self.get_parameter("rerun_save").value or None
        rerun_connect = self.get_parameter("rerun_connect").value or None
        if rerun_save and rerun_connect:
            raise ValueError("rerun_save and rerun_connect are mutually exclusive")
        if (rerun_save or rerun_connect) and viewer is None:
            viewer = "rerun"

        return DPVO(
            self.cfg,
            self.get_parameter("network").value,
            ht=image_shape[0],
            wd=image_shape[1],
            viewer=viewer,
            viewer_output=Path(rerun_save) if rerun_save else None,
            viewer_connect=rerun_connect,
            viewer_recording_id=(
                self.get_parameter("rerun_recording_id").value or None
            ),
            viewer_entity_prefix=(
                self.get_parameter("rerun_entity_prefix").value or None
            ),
            long_term_lc_factory=loop_closure_factory,
        )

    def _publish_pose(self, image_message):
        if self.slam.n <= 0 or self.slam.n == self.last_published_keyframe:
            return
        self.last_published_keyframe = self.slam.n
        poses = (
            SE3(self.slam.pg.poses_[: self.slam.n])
            .inv()
            .data.detach()
            .float()
            .cpu()
            .numpy()
        )
        poses = transform_poses_xyzw(
            self.slam.pg.session_from_map_,
            poses,
        )
        pose_data = poses[-1]
        pose = PoseStamped()
        pose.header.stamp = image_message.header.stamp
        pose.header.frame_id = self.get_parameter("map_frame").value
        pose.pose.position.x = float(pose_data[0])
        pose.pose.position.y = float(pose_data[1])
        pose.pose.position.z = float(pose_data[2])
        pose.pose.orientation.x = float(pose_data[3])
        pose.pose.orientation.y = float(pose_data[4])
        pose.pose.orientation.z = float(pose_data[5])
        pose.pose.orientation.w = float(pose_data[6])
        self.pose_publisher.publish(pose)

        # Rebuild the path because DPVO bundle adjustment updates historical
        # poses. All poses are expressed in the stable session map gauge.
        self.path = PathMessage()
        self.path.header.stamp = pose.header.stamp
        self.path.header.frame_id = self.get_parameter("map_frame").value
        for pose_data in poses:
            path_pose = PoseStamped()
            path_pose.header = self.path.header
            path_pose.pose.position.x = float(pose_data[0])
            path_pose.pose.position.y = float(pose_data[1])
            path_pose.pose.position.z = float(pose_data[2])
            path_pose.pose.orientation.x = float(pose_data[3])
            path_pose.pose.orientation.y = float(pose_data[4])
            path_pose.pose.orientation.z = float(pose_data[5])
            path_pose.pose.orientation.w = float(pose_data[6])
            self.path.poses.append(path_pose)
        self.path_publisher.publish(self.path)

    def close(self):
        if self.slam is not None:
            with self.processing_lock:
                # Three concurrent final DPVO optimization sweeps exceed the
                # shared GPU's remaining memory. Online BA has already run;
                # ROS shutdown only needs backend cleanup and Rerun flushing.
                self.slam.terminate(final_updates=0)
                self.slam = None


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotDpvoNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

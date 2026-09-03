from __future__ import annotations

import re
import threading

import numpy as np
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from dpvo.loop_closure.distributed import (
    BowMatch,
    FrameIdentity,
    GlobalDescriptor,
    InterRobotConstraint,
    KeyframePayload,
    SparseBow,
)
from dpvo_multi_robot_interfaces.msg import (
    BowVector,
    GlobalDescriptor as GlobalDescriptorMessage,
    InterRobotLoopClosure,
    KeyframeData,
)
from dpvo_multi_robot_interfaces.srv import GetKeyframe


def _stamp_from_seconds(value):
    seconds = int(value)
    nanoseconds = int(round((value - seconds) * 1e9))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    from builtin_interfaces.msg import Time

    return Time(sec=seconds, nanosec=nanoseconds)


class Ros2DistributedTransport:
    def __init__(
        self,
        node,
        robot_id: str,
        processing_lock: threading.RLock,
        bow_topic: str = "/dpvo_multi_robot/bow",
        global_descriptor_topic: str = "/dpvo_multi_robot/global_descriptor",
        constraint_topic: str = "/dpvo_multi_robot/loop_closure",
        service_prefix: str = "/dpvo_multi_robot",
    ):
        if not re.fullmatch(r"[A-Za-z0-9_]+", robot_id):
            raise ValueError("robot_id may contain only letters, numbers, and underscores")
        self.node = node
        self.robot_id = robot_id
        self.processing_lock = processing_lock
        self.service_prefix = service_prefix.rstrip("/")
        self.backend = None
        self.clients = {}
        self.callback_group = ReentrantCallbackGroup()
        # Detailed keyframe construction runs the local feature front end and
        # triangulation while holding the DPVO processing lock. Scheduling
        # several services concurrently would leave executor threads blocked
        # on that lock and can starve the live image callback. BoW and client
        # traffic remain reentrant; only heavyweight exports are serialized.
        self.keyframe_service_group = MutuallyExclusiveCallbackGroup()

        bow_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=500,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.bow_publisher = node.create_publisher(BowVector, bow_topic, bow_qos)
        self.bow_subscription = node.create_subscription(
            BowVector,
            bow_topic,
            self._receive_bow,
            bow_qos,
            callback_group=self.callback_group,
        )
        self.global_descriptor_publisher = node.create_publisher(
            GlobalDescriptorMessage,
            global_descriptor_topic,
            bow_qos,
        )
        self.global_descriptor_subscription = node.create_subscription(
            GlobalDescriptorMessage,
            global_descriptor_topic,
            self._receive_global_descriptor,
            bow_qos,
            callback_group=self.callback_group,
        )
        self.constraint_publisher = node.create_publisher(
            InterRobotLoopClosure,
            constraint_topic,
            20,
        )
        self.keyframe_service = node.create_service(
            GetKeyframe,
            self._service_name(robot_id),
            self._get_keyframe,
            callback_group=self.keyframe_service_group,
        )

    def _service_name(self, robot_id):
        return f"{self.service_prefix}/{robot_id}/get_keyframe"

    def bind(self, backend):
        self.backend = backend

    def log(self, message):
        self.node.get_logger().info(message)

    def publish_bow(self, frame, bow, vocabulary_id, timestamp):
        if not self.node.context.ok():
            return
        message = BowVector()
        message.header.stamp = _stamp_from_seconds(timestamp)
        message.header.frame_id = frame.robot_id
        message.robot_id = frame.robot_id
        message.session_id = frame.session_id
        message.keyframe_id = frame.keyframe_id
        message.vocabulary_id = vocabulary_id
        message.word_ids = bow.word_ids.tolist()
        message.word_values = bow.word_values.tolist()
        self.bow_publisher.publish(message)

    def _receive_bow(self, message):
        if self.backend is None:
            return
        try:
            self.backend.receive_bow(
                FrameIdentity(
                    message.robot_id,
                    message.session_id,
                    int(message.keyframe_id),
                ),
                SparseBow(message.word_ids, message.word_values),
                message.vocabulary_id,
            )
        except (TypeError, ValueError) as error:
            self.node.get_logger().warning(f"Rejected malformed BoW message: {error}")

    def publish_global_descriptor(self, frame, descriptor, timestamp):
        if not self.node.context.ok():
            return
        message = GlobalDescriptorMessage()
        message.header.stamp = _stamp_from_seconds(timestamp)
        message.header.frame_id = frame.robot_id
        message.robot_id = frame.robot_id
        message.session_id = frame.session_id
        message.keyframe_id = frame.keyframe_id
        message.model_id = descriptor.model_id
        message.dimension = descriptor.values.size
        message.values = descriptor.values.tolist()
        self.global_descriptor_publisher.publish(message)

    def _receive_global_descriptor(self, message):
        if self.backend is None:
            return
        try:
            values = np.asarray(message.values, dtype=np.float32)
            if values.size != int(message.dimension):
                raise ValueError(
                    "global descriptor dimension does not match its payload"
                )
            self.backend.receive_global_descriptor(
                FrameIdentity(
                    message.robot_id,
                    message.session_id,
                    int(message.keyframe_id),
                ),
                GlobalDescriptor(values, message.model_id),
            )
        except (TypeError, ValueError) as error:
            self.node.get_logger().warning(
                f"Rejected malformed global descriptor message: {error}"
            )

    def request_keyframe(self, match: BowMatch):
        if not self.node.context.ok():
            return
        robot_id = match.remote_frame.robot_id
        client = self.clients.get(robot_id)
        if client is None:
            client = self.node.create_client(
                GetKeyframe,
                self._service_name(robot_id),
                callback_group=self.callback_group,
            )
            self.clients[robot_id] = client
        if not client.service_is_ready():
            self.backend.receive_keyframe(match, None, "remote keyframe service unavailable")
            return

        request = GetKeyframe.Request()
        request.robot_id = robot_id
        request.session_id = match.remote_frame.session_id
        request.keyframe_id = match.remote_frame.keyframe_id
        future = client.call_async(request)
        future.add_done_callback(lambda completed: self._keyframe_response(match, completed))

    def _keyframe_response(self, match, future):
        try:
            response = future.result()
            if response is None or not response.success:
                error = "empty service response" if response is None else response.error
                self.backend.receive_keyframe(match, None, error)
                return
            self.backend.receive_keyframe(
                match,
                self._payload_from_message(response.keyframe),
            )
        except Exception as error:
            self.backend.receive_keyframe(match, None, str(error))

    def _get_keyframe(self, request, response):
        if self.backend is None:
            response.success = False
            response.error = "DPVO backend is not initialized"
            return response
        frame = FrameIdentity(
            request.robot_id,
            request.session_id,
            int(request.keyframe_id),
        )
        try:
            with self.processing_lock:
                payload = self.backend.export_keyframe(frame)
            response.keyframe = self._message_from_payload(payload)
            response.success = True
        except Exception as error:
            response.success = False
            response.error = str(error)
        return response

    @staticmethod
    def _message_from_payload(payload):
        message = KeyframeData()
        message.header.stamp = _stamp_from_seconds(payload.timestamp)
        message.header.frame_id = payload.frame.robot_id
        message.robot_id = payload.frame.robot_id
        message.session_id = payload.frame.session_id
        message.keyframe_id = payload.frame.keyframe_id
        message.timestamp = payload.timestamp
        message.pose = np.asarray(payload.pose, dtype=np.float32).reshape(7).tolist()
        message.point_count = payload.points.shape[0]
        message.descriptor_dim = payload.descriptors.shape[1]
        message.image_width = int(payload.image_size[0])
        message.image_height = int(payload.image_size[1])
        message.points_xyz = np.asarray(payload.points, dtype=np.float32).reshape(-1).tolist()
        message.keypoints_uv = np.asarray(payload.keypoints, dtype=np.float32).reshape(-1).tolist()
        message.descriptors = (
            np.asarray(payload.descriptors, dtype=np.float32).reshape(-1).tolist()
        )
        return message

    @staticmethod
    def _payload_from_message(message):
        count = int(message.point_count)
        descriptor_dim = int(message.descriptor_dim)
        points = np.asarray(message.points_xyz, dtype=np.float32)
        keypoints = np.asarray(message.keypoints_uv, dtype=np.float32)
        descriptors = np.asarray(message.descriptors, dtype=np.float32)
        if points.size != count * 3 or keypoints.size != count * 2:
            raise ValueError("remote keyframe point/keypoint dimensions are inconsistent")
        if descriptors.size != count * descriptor_dim:
            raise ValueError("remote keyframe descriptor dimensions are inconsistent")
        return KeyframePayload(
            frame=FrameIdentity(
                message.robot_id,
                message.session_id,
                int(message.keyframe_id),
            ),
            timestamp=float(message.timestamp),
            pose=np.asarray(message.pose, dtype=np.float32),
            points=points.reshape(count, 3),
            keypoints=keypoints.reshape(count, 2),
            descriptors=descriptors.reshape(count, descriptor_dim),
            image_size=np.array(
                [message.image_width, message.image_height],
                dtype=np.float32,
            ),
        )

    def publish_constraint(self, constraint: InterRobotConstraint):
        if not self.node.context.ok():
            return
        message = InterRobotLoopClosure()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = constraint.query_frame.robot_id
        message.query_robot_id = constraint.query_frame.robot_id
        message.query_session_id = constraint.query_frame.session_id
        message.query_keyframe_id = constraint.query_frame.keyframe_id
        message.match_robot_id = constraint.match_frame.robot_id
        message.match_session_id = constraint.match_frame.session_id
        message.match_keyframe_id = constraint.match_frame.keyframe_id
        message.bow_score = constraint.bow_score
        message.query_pose = np.asarray(
            constraint.query_pose,
            dtype=np.float32,
        ).reshape(7).tolist()
        message.match_pose = np.asarray(
            constraint.match_pose,
            dtype=np.float32,
        ).reshape(7).tolist()
        message.query_to_match.translation.x = float(constraint.translation[0])
        message.query_to_match.translation.y = float(constraint.translation[1])
        message.query_to_match.translation.z = float(constraint.translation[2])
        message.query_to_match.rotation.x = float(constraint.quaternion_xyzw[0])
        message.query_to_match.rotation.y = float(constraint.quaternion_xyzw[1])
        message.query_to_match.rotation.z = float(constraint.quaternion_xyzw[2])
        message.query_to_match.rotation.w = float(constraint.quaternion_xyzw[3])
        message.scale = constraint.scale
        message.inliers = constraint.inliers
        message.inlier_ratio = constraint.inlier_ratio
        message.verification_method = constraint.verification_method
        self.constraint_publisher.publish(message)
        self.node.get_logger().info(
            "Inter-robot loop %s:%d -> %s:%d: %.3f retrieval, %d inliers, "
            "scale %.4f, %s"
            % (
                constraint.query_frame.robot_id,
                constraint.query_frame.keyframe_id,
                constraint.match_frame.robot_id,
                constraint.match_frame.keyframe_id,
                constraint.bow_score,
                constraint.inliers,
                constraint.scale,
                constraint.verification_method,
            )
        )

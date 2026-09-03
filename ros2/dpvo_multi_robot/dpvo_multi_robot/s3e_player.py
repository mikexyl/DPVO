from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header

from .cu_multi_core import build_pinhole_rectification, rectified_camera_info
from .newer_college_core import PlaybackState, stamp_components
from .s3e_core import decode_compressed_bgr8, read_s3e_calibration


class S3EBagPlayer(Node):
    """Play one S3E robot's compressed left camera with DPVO backpressure."""

    def __init__(self):
        super().__init__("s3e_player")
        self.declare_parameter("bag", "")
        self.declare_parameter("calib", "")
        self.declare_parameter("source_topic", "/Alpha/left_camera/compressed")
        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("camera_info_topic", "camera/camera_info")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("done_topic", "dpvo/player_done")
        self.declare_parameter("exit_on_finish", False)
        self.declare_parameter("stride", 2)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("rectification_alpha", 0.0)

        bag = Path(self.get_parameter("bag").value).expanduser()
        if not bag.is_dir():
            raise FileNotFoundError(f"S3E ROS2 bag does not exist: {bag}")
        calibration = read_s3e_calibration(
            Path(self.get_parameter("calib").value)
        )
        self.rectification = build_pinhole_rectification(
            calibration.camera,
            float(self.get_parameter("rectification_alpha").value),
        )
        self.info_fields = rectified_camera_info(self.rectification)
        self.playback = PlaybackState(
            int(self.get_parameter("stride").value),
            int(self.get_parameter("start_frame").value),
            int(self.get_parameter("max_frames").value),
        )

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.image_publisher = self.create_publisher(
            Image, self.get_parameter("image_topic").value, reliable
        )
        self.info_publisher = self.create_publisher(
            CameraInfo, self.get_parameter("camera_info_topic").value, latched
        )
        self.done_publisher = self.create_publisher(
            Bool, self.get_parameter("done_topic").value, latched
        )
        self.ack_subscription = self.create_subscription(
            Header,
            self.get_parameter("frame_ack_topic").value,
            self._acknowledge,
            latched,
        )

        # S3Ev1 bags predate embedded rosbag2 message definitions. All topics
        # used here are standard ROS 2 interfaces, so provide the Humble
        # typestore explicitly instead of asking AnyReader to infer definitions.
        self.reader = AnyReader(
            [bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)
        )
        self.reader.open()
        source_topic = str(self.get_parameter("source_topic").value)
        self.connections = [
            connection
            for connection in self.reader.connections
            if connection.topic == source_topic
        ]
        if len(self.connections) != 1:
            self.reader.close()
            raise RuntimeError(
                f"expected one {source_topic} connection in {bag}, "
                f"found {len(self.connections)}"
            )
        connection = self.connections[0]
        if not connection.msgtype.endswith("/CompressedImage"):
            self.reader.close()
            raise TypeError(
                f"{source_topic} is {connection.msgtype}, expected CompressedImage"
            )
        self.messages = self.reader.messages(connections=self.connections)
        self.calibration_announced = False
        self.shutdown_timer = None
        self.timer = self.create_timer(0.01, self._publish_next)
        matrix = self.rectification.camera_matrix
        self.get_logger().info(
            f"Playing {source_topic} at stride {self.playback.stride} with DPVO "
            f"backpressure; rectified K=[{matrix[0, 0]:.6g}, "
            f"{matrix[1, 1]:.6g}, {matrix[0, 2]:.6g}, {matrix[1, 2]:.6g}]"
        )

    def _acknowledge(self, _message):
        if self.playback.acknowledge() and not self.playback.finished:
            self._finish()

    def _publish_next(self):
        if self.playback.waiting_for_ack or self.playback.finished:
            return
        if self.playback.limit_reached:
            self._finish()
            return
        if (
            self.image_publisher.get_subscription_count() == 0
            or self.info_publisher.get_subscription_count() == 0
        ):
            return
        if not self.calibration_announced:
            # Give the tracker one executor turn to consume the transient-local
            # calibration before the first image. Otherwise the image callback
            # can run first and acknowledge one frame without tracking it.
            info = CameraInfo()
            info.header.frame_id = "left_camera"
            info.height = self.info_fields["height"]
            info.width = self.info_fields["width"]
            info.k = self.info_fields["k"]
            info.r = self.info_fields["r"]
            info.p = self.info_fields["p"]
            info.d = self.info_fields["d"]
            info.distortion_model = self.info_fields["distortion_model"]
            self.info_publisher.publish(info)
            self.calibration_announced = True
            return

        while True:
            try:
                connection, _bag_timestamp, rawdata = next(self.messages)
            except StopIteration:
                self._finish()
                return
            if self.playback.select_next_source():
                break

        source = self.reader.deserialize(rawdata, connection.msgtype)
        source_format = str(source.format).lower()
        if "jpeg" not in source_format and "jpg" not in source_format:
            raise ValueError(
                f"S3E compressed format must be JPEG, got {source.format!r}"
            )
        image_data = self.rectification.rectify(
            decode_compressed_bgr8(bytes(source.data))
        )
        stamp_sec, stamp_nanosec = stamp_components(source.header.stamp)

        image = Image()
        image.header.stamp.sec = stamp_sec
        image.header.stamp.nanosec = stamp_nanosec
        image.header.frame_id = str(source.header.frame_id) or "left_camera"
        image.height, image.width = image_data.shape[:2]
        image.encoding = "bgr8"
        image.is_bigendian = 0
        image.step = int(image.width * 3)
        image.data = image_data.tobytes()

        info = CameraInfo()
        info.header = image.header
        info.height = self.info_fields["height"]
        info.width = self.info_fields["width"]
        info.k = self.info_fields["k"]
        info.r = self.info_fields["r"]
        info.p = self.info_fields["p"]
        info.d = self.info_fields["d"]
        info.distortion_model = self.info_fields["distortion_model"]

        self.info_publisher.publish(info)
        self.image_publisher.publish(image)
        self.playback.mark_published()
        if self.playback.processed % 100 == 0:
            self.get_logger().info(
                f"Published {self.playback.processed} frames; "
                f"source frame {self.playback.source_index}"
            )

    def _finish(self):
        if self.playback.finished:
            return
        self.playback.finish()
        self.reader.close()
        self.done_publisher.publish(Bool(data=True))
        self.get_logger().info(
            f"Finished after {self.playback.processed} acknowledged frames"
        )
        if self.get_parameter("exit_on_finish").value:
            self.shutdown_timer = self.create_timer(0.5, self._shutdown)

    def _shutdown(self):
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
        if self.context.ok():
            rclpy.shutdown(context=self.context)


def main(args=None):
    rclpy.init(args=args)
    node = S3EBagPlayer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not node.playback.finished:
            node.reader.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

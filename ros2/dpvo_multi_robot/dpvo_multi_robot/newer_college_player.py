from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosbags.highlevel import AnyReader
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header

from .newer_college_core import (
    PlaybackState,
    build_fisheye_rectification,
    decode_compressed_mono8,
    read_kalibr_camera_calibration,
    rectified_camera_info,
    stamp_components,
)


class NewerCollegeBagPlayer(Node):
    """Rectify Newer College ROS1 cam0 with per-frame DPVO backpressure."""

    def __init__(self):
        super().__init__("newer_college_player")
        self.declare_parameter("bag", "")
        self.declare_parameter("calib", "")
        self.declare_parameter("camera_key", "cam0")
        self.declare_parameter(
            "source_topic", "/alphasense_driver_ros/cam0/compressed"
        )
        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("camera_info_topic", "camera/camera_info")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("done_topic", "dpvo/player_done")
        self.declare_parameter("exit_on_finish", False)
        self.declare_parameter("stride", 2)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("rectification_balance", 0.0)

        bag = Path(self.get_parameter("bag").value).expanduser()
        if not bag.is_file():
            raise FileNotFoundError(f"Newer College bag does not exist: {bag}")
        calibration = read_kalibr_camera_calibration(
            Path(self.get_parameter("calib").value),
            str(self.get_parameter("camera_key").value),
        )
        self.rectification = build_fisheye_rectification(
            calibration,
            float(self.get_parameter("rectification_balance").value),
        )
        self.info_fields = rectified_camera_info(self.rectification)
        self.playback = PlaybackState(
            int(self.get_parameter("stride").value),
            int(self.get_parameter("start_frame").value),
            int(self.get_parameter("max_frames").value),
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.image_publisher = self.create_publisher(
            Image, self.get_parameter("image_topic").value, image_qos
        )
        self.info_publisher = self.create_publisher(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            latched_qos,
        )
        self.done_publisher = self.create_publisher(
            Bool, self.get_parameter("done_topic").value, latched_qos
        )
        self.ack_subscription = self.create_subscription(
            Header,
            self.get_parameter("frame_ack_topic").value,
            self._acknowledge,
            latched_qos,
        )

        self.reader = AnyReader([bag])
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
                f"{source_topic} in {bag} is {connection.msgtype}, "
                "expected sensor_msgs/CompressedImage"
            )
        self.messages = self.reader.messages(connections=self.connections)
        self.shutdown_timer = None
        self.timer = self.create_timer(0.01, self._publish_next)
        new_k = self.rectification.camera_matrix
        self.get_logger().info(
            f"Playing {bag.name} cam0 at stride {self.playback.stride} with "
            f"DPVO backpressure; rectified K="
            f"[{new_k[0, 0]:.6g}, {new_k[1, 1]:.6g}, "
            f"{new_k[0, 2]:.6g}, {new_k[1, 2]:.6g}]"
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

        while True:
            try:
                connection, _bag_timestamp_ns, rawdata = next(self.messages)
            except StopIteration:
                self._finish()
                return
            if self.playback.select_next_source():
                break

        source = self.reader.deserialize(rawdata, connection.msgtype)
        source_format = str(source.format).lower()
        if "jpeg" not in source_format and "jpg" not in source_format:
            raise ValueError(
                f"cam0 compressed format must be JPEG, got {source.format!r}"
            )
        image_data = self.rectification.rectify(
            decode_compressed_mono8(bytes(source.data))
        )
        stamp_sec, stamp_nanosec = stamp_components(source.header.stamp)

        image = Image()
        # Deliberately preserve the camera header stamp. Bag record time is not
        # a camera measurement timestamp and is never used here.
        image.header.stamp.sec = stamp_sec
        image.header.stamp.nanosec = stamp_nanosec
        image.header.frame_id = str(source.header.frame_id) or "cam0"
        image.height, image.width = image_data.shape
        image.encoding = "mono8"
        image.is_bigendian = 0
        image.step = int(image.width)
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
    node = NewerCollegeBagPlayer()
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

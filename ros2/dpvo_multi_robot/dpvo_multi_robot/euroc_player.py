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


class EurocBagPlayer(Node):
    """Play a ROS 1 EuRoC bag with per-frame DPVO acknowledgement."""

    def __init__(self):
        super().__init__("euroc_player")
        self.declare_parameter("bag", "")
        self.declare_parameter("calib", "calib/euroc.txt")
        self.declare_parameter("source_topic", "/cam0/image_raw")
        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("camera_info_topic", "camera/camera_info")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("done_topic", "dpvo/player_done")
        self.declare_parameter("stride", 2)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("max_frames", 0)

        bag = Path(self.get_parameter("bag").value)
        if not bag.is_file():
            raise FileNotFoundError(f"EuRoC bag does not exist: {bag}")
        self.stride = int(self.get_parameter("stride").value)
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.start_frame = int(self.get_parameter("start_frame").value)
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        self.calibration = np.loadtxt(self.get_parameter("calib").value)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        done_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        ack_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.image_publisher = self.create_publisher(
            Image,
            self.get_parameter("image_topic").value,
            image_qos,
        )
        self.info_publisher = self.create_publisher(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            info_qos,
        )
        self.done_publisher = self.create_publisher(
            Bool,
            self.get_parameter("done_topic").value,
            done_qos,
        )
        ack_topic = self.get_parameter("frame_ack_topic").value
        self.ack_subscription = self.create_subscription(
            Header,
            ack_topic,
            self._acknowledge,
            ack_qos,
        )

        self.reader = AnyReader([bag])
        self.reader.open()
        source_topic = self.get_parameter("source_topic").value
        self.connections = [
            connection
            for connection in self.reader.connections
            if connection.topic == source_topic
        ]
        if not self.connections:
            self.reader.close()
            raise RuntimeError(f"No {source_topic} connection in {bag}")
        self.messages = self.reader.messages(connections=self.connections)
        self.source_index = -1
        self.selected_index = -1
        self.processed = 0
        self.waiting_for_ack = False
        self.finished = False
        self.timer = self.create_timer(0.01, self._publish_next)
        self.get_logger().info(
            f"Playing {bag.name} with stride {self.stride} and DPVO backpressure"
        )

    def _acknowledge(self, _message):
        self.waiting_for_ack = False

    def _publish_next(self):
        if self.waiting_for_ack or self.finished:
            return
        if (
            self.image_publisher.get_subscription_count() == 0
            or self.info_publisher.get_subscription_count() == 0
        ):
            return
        while True:
            try:
                connection, timestamp_ns, rawdata = next(self.messages)
            except StopIteration:
                self._finish()
                return
            self.source_index += 1
            if self.source_index % self.stride == 0:
                self.selected_index += 1
                if self.selected_index >= self.start_frame:
                    break

        source = self.reader.deserialize(rawdata, connection.msgtype)
        image = Image()
        image.header.stamp.sec = timestamp_ns // 1_000_000_000
        image.header.stamp.nanosec = timestamp_ns % 1_000_000_000
        image.header.frame_id = "cam0"
        image.height = int(source.height)
        image.width = int(source.width)
        image.encoding = source.encoding
        image.is_bigendian = int(source.is_bigendian)
        image.step = int(source.step)
        image.data = bytes(source.data)

        fx, fy, cx, cy = self.calibration[:4]
        info = CameraInfo()
        info.header = image.header
        info.height = int(source.height)
        info.width = int(source.width)
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = self.calibration[4:].tolist()
        info.distortion_model = "plumb_bob"
        self.info_publisher.publish(info)
        self.image_publisher.publish(image)
        self.waiting_for_ack = True
        self.processed += 1
        if self.processed % 100 == 0:
            self.get_logger().info(f"Published and acknowledged {self.processed} frames")
        if self.max_frames > 0 and self.processed >= self.max_frames:
            self._finish()

    def _finish(self):
        if self.finished:
            return
        self.finished = True
        self.reader.close()
        self.done_publisher.publish(Bool(data=True))
        self.get_logger().info(f"Finished after {self.processed} frames")


def main(args=None):
    rclpy.init(args=args)
    node = EurocBagPlayer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not node.finished:
            node.reader.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

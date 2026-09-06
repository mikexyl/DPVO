from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header

from .scalemaster_core import read_camera_matrix, read_frames


class ScaleMasterPlayer(Node):
    """Play one ScaleMaster image sequence with DPVO backpressure."""

    def __init__(self):
        super().__init__("scalemaster_player")
        self.declare_parameter("sequence_dir", "")
        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("camera_info_topic", "camera/camera_info")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("done_topic", "dpvo/player_done")
        self.declare_parameter("exit_on_finish", False)
        self.declare_parameter("stride", 5)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("max_frames", 0)

        sequence_dir = Path(self.get_parameter("sequence_dir").value).expanduser()
        if not sequence_dir.is_dir():
            raise FileNotFoundError(
                f"ScaleMaster sequence directory does not exist: {sequence_dir}"
            )
        all_frames = read_frames(sequence_dir)
        self.camera_matrix = read_camera_matrix(sequence_dir / "camera_matrix.csv")

        stride = int(self.get_parameter("stride").value)
        start_frame = int(self.get_parameter("start_frame").value)
        max_frames = int(self.get_parameter("max_frames").value)
        if stride < 1:
            raise ValueError("stride must be positive")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        self.frames = all_frames[::stride][start_frame:]
        if max_frames:
            self.frames = self.frames[:max_frames]
        if not self.frames:
            raise ValueError("ScaleMaster frame selection is empty")

        sample = cv2.imread(str(self.frames[0].image_path), cv2.IMREAD_COLOR)
        if sample is None:
            raise RuntimeError(
                f"failed to decode ScaleMaster image: {self.frames[0].image_path}"
            )
        self.height, self.width = sample.shape[:2]

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

        self.next_frame = 0
        self.waiting_for_ack = False
        self.finished = False
        self.calibration_announced = False
        self.first_image_not_before = None
        self.shutdown_timer = None
        self.timer = self.create_timer(0.01, self._publish_next)
        self.get_logger().info(
            f"Playing {sequence_dir.name}: {len(self.frames)} of {len(all_frames)} "
            f"RGB frames at stride {stride} with DPVO backpressure"
        )

    def _camera_info(self, image_header=None):
        matrix = self.camera_matrix
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        info = CameraInfo()
        if image_header is not None:
            info.header = image_header
        else:
            info.header.frame_id = "camera"
        info.height = self.height
        info.width = self.width
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = []
        info.distortion_model = "plumb_bob"
        return info

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
        if not self.calibration_announced:
            self.info_publisher.publish(self._camera_info())
            self.calibration_announced = True
            # The tracker acknowledges images even when CameraInfo has not yet
            # reached its callback. The publisher/subscriber counts can become
            # nonzero before that transient-local sample is consumed, so leave
            # an explicit discovery/callback interval before the first image.
            self.first_image_not_before = time.monotonic() + 0.5
            return
        if time.monotonic() < self.first_image_not_before:
            return
        if self.next_frame >= len(self.frames):
            self._finish()
            return

        frame = self.frames[self.next_frame]
        image_data = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image_data is None:
            raise RuntimeError(f"failed to decode ScaleMaster image: {frame.image_path}")
        if image_data.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"ScaleMaster image {frame.image_path} has shape "
                f"{image_data.shape[:2]}, expected {(self.height, self.width)}"
            )
        image_data = np.ascontiguousarray(image_data)
        timestamp_ns = int(round(frame.timestamp * 1_000_000_000))

        image = Image()
        image.header.stamp.sec = timestamp_ns // 1_000_000_000
        image.header.stamp.nanosec = timestamp_ns % 1_000_000_000
        image.header.frame_id = "camera"
        image.height = self.height
        image.width = self.width
        image.encoding = "bgr8"
        image.is_bigendian = 0
        image.step = self.width * 3
        image.data = image_data.tobytes()

        self.info_publisher.publish(self._camera_info(image.header))
        self.image_publisher.publish(image)
        self.waiting_for_ack = True
        self.next_frame += 1
        if self.next_frame % 100 == 0:
            self.get_logger().info(
                f"Published {self.next_frame} frames; source frame {frame.frame_id}"
            )

    def _finish(self):
        if self.finished:
            return
        self.finished = True
        self.done_publisher.publish(Bool(data=True))
        self.get_logger().info(f"Finished after {self.next_frame} acknowledged frames")
        if self.get_parameter("exit_on_finish").value:
            self.shutdown_timer = self.create_timer(0.5, self._shutdown)

    def _shutdown(self):
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
        if self.context.ok():
            rclpy.shutdown(context=self.context)


def main(args=None):
    rclpy.init(args=args)
    node = ScaleMasterPlayer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

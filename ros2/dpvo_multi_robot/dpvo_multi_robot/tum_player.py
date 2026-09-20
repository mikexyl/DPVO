from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
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
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header


@dataclass(frozen=True)
class TumRgbFrame:
    timestamp: float
    image_path: Path


def read_tum_rgb_list(sequence_dir: Path, rgb_list: str = "rgb.txt"):
    """Read timestamp/image pairs from a TUM RGB-D rgb.txt file."""

    list_path = sequence_dir / rgb_list
    if not list_path.is_file():
        raise FileNotFoundError(f"TUM RGB list does not exist: {list_path}")

    frames = []
    for line_number, line in enumerate(list_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed {list_path}:{line_number}: {line}")
        image_path = sequence_dir / fields[1]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"TUM RGB image from {list_path}:{line_number} does not exist: "
                f"{image_path}"
            )
        frames.append(TumRgbFrame(float(fields[0]), image_path))

    if not frames:
        raise RuntimeError(f"No RGB frames listed in {list_path}")
    return frames


class TumRgbPlayer(Node):
    """Play a TUM RGB-D image sequence with per-frame DPVO acknowledgement."""

    def __init__(self, **node_options):
        super().__init__("tum_player", **node_options)
        self.declare_parameter("sequence_dir", "")
        self.declare_parameter("rgb_list", "rgb.txt")
        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("camera_info_topic", "camera/camera_info")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("done_topic", "dpvo/player_done")
        self.declare_parameter("exit_on_finish", False)
        self.declare_parameter("stride", 1)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("crop_x", 16)
        self.declare_parameter("crop_y", 8)
        self.declare_parameter("fx", 517.3)
        self.declare_parameter("fy", 516.5)
        self.declare_parameter("cx", 318.6)
        self.declare_parameter("cy", 255.3)
        self.declare_parameter("k1", 0.2624)
        self.declare_parameter("k2", -0.9531)
        self.declare_parameter("p1", -0.0054)
        self.declare_parameter("p2", 0.0026)
        self.declare_parameter("k3", 1.1633)

        sequence_dir = Path(self.get_parameter("sequence_dir").value)
        if not sequence_dir.is_dir():
            raise FileNotFoundError(
                f"TUM RGB-D sequence directory does not exist: {sequence_dir}"
            )
        self.frames = read_tum_rgb_list(
            sequence_dir,
            self.get_parameter("rgb_list").value,
        )

        self.stride = int(self.get_parameter("stride").value)
        self.start_frame = int(self.get_parameter("start_frame").value)
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.crop_x = int(self.get_parameter("crop_x").value)
        self.crop_y = int(self.get_parameter("crop_y").value)
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.crop_x < 0 or self.crop_y < 0:
            raise ValueError("crop values must be non-negative")

        self.camera_matrix = np.array(
            [
                [
                    float(self.get_parameter("fx").value),
                    0.0,
                    float(self.get_parameter("cx").value),
                ],
                [
                    0.0,
                    float(self.get_parameter("fy").value),
                    float(self.get_parameter("cy").value),
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.distortion = np.array(
            [
                float(self.get_parameter("k1").value),
                float(self.get_parameter("k2").value),
                float(self.get_parameter("p1").value),
                float(self.get_parameter("p2").value),
                float(self.get_parameter("k3").value),
            ],
            dtype=np.float64,
        )

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
        self.ack_subscription = self.create_subscription(
            Header,
            self.get_parameter("frame_ack_topic").value,
            self._acknowledge,
            ack_qos,
        )

        self.selected_frames = self.frames[:: self.stride][self.start_frame :]
        if self.max_frames > 0:
            self.selected_frames = self.selected_frames[: self.max_frames]
        self.next_frame = 0
        self.waiting_for_ack = False
        self.finished = False
        self.shutdown_timer = None
        self.timer = self.create_timer(0.01, self._publish_next)
        self.get_logger().info(
            f"Playing {sequence_dir.name}: {len(self.selected_frames)} of "
            f"{len(self.frames)} RGB frames, stride {self.stride}, with DPVO "
            "backpressure"
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
        if self.next_frame >= len(self.selected_frames):
            self._finish()
            return

        frame = self.selected_frames[self.next_frame]
        image_data = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image_data is None:
            raise RuntimeError(f"Failed to decode TUM RGB image: {frame.image_path}")
        image_data = cv2.undistort(
            image_data,
            self.camera_matrix,
            self.distortion,
        )
        if self.crop_y:
            image_data = image_data[self.crop_y : -self.crop_y]
        if self.crop_x:
            image_data = image_data[:, self.crop_x : -self.crop_x]
        image_data = np.ascontiguousarray(image_data)

        timestamp_ns = int(round(frame.timestamp * 1_000_000_000))
        image = Image()
        image.header.stamp.sec = timestamp_ns // 1_000_000_000
        image.header.stamp.nanosec = timestamp_ns % 1_000_000_000
        image.header.frame_id = "cam0"
        image.height, image.width = image_data.shape[:2]
        image.encoding = "bgr8"
        image.is_bigendian = 0
        image.step = int(image.width * 3)
        image.data = image_data.tobytes()

        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2]) - self.crop_x
        cy = float(self.camera_matrix[1, 2]) - self.crop_y
        info = CameraInfo()
        info.header = image.header
        info.height = image.height
        info.width = image.width
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = []
        info.distortion_model = "plumb_bob"

        self.info_publisher.publish(info)
        self.image_publisher.publish(image)
        self.waiting_for_ack = True
        self.next_frame += 1
        if self.next_frame % 100 == 0:
            self.get_logger().info(
                f"Published and acknowledged {self.next_frame} frames"
            )

    def _finish(self):
        if self.finished:
            return
        self.finished = True
        self.done_publisher.publish(Bool(data=True))
        self.get_logger().info(f"Finished after {self.next_frame} frames")
        if self.get_parameter("exit_on_finish").value:
            # Give the transient-local completion sample time to reach the
            # tracker before this short-lived player process exits.
            self.shutdown_timer = self.create_timer(0.5, self._shutdown)

    def _shutdown(self):
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
        if self.context.ok():
            rclpy.shutdown(context=self.context)


def main(args=None):
    rclpy.init(args=args)
    node = TumRgbPlayer()
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

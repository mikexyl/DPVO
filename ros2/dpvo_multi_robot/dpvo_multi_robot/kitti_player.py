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
class KittiFrame:
    index: int
    timestamp: float
    image_path: Path


def read_kitti_calibration(path: Path) -> dict[str, np.ndarray]:
    calibration = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Malformed {path}:{line_number}: {line}")
        key, values = line.split(":", 1)
        calibration[key] = np.asarray(
            [float(value) for value in values.split()], dtype=np.float64
        )
    return calibration


def read_kitti_frames(sequence_dir: Path, image_dir: str) -> list[KittiFrame]:
    times_path = sequence_dir / "times.txt"
    calibration_path = sequence_dir / "calib.txt"
    images_path = sequence_dir / image_dir
    if not times_path.is_file():
        raise FileNotFoundError(f"KITTI timestamps do not exist: {times_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"KITTI calibration does not exist: {calibration_path}"
        )
    if not images_path.is_dir():
        raise FileNotFoundError(f"KITTI image directory does not exist: {images_path}")

    timestamps = [
        float(line.strip())
        for line in times_path.read_text().splitlines()
        if line.strip()
    ]
    images = sorted(images_path.glob("*.png"))
    if len(images) != len(timestamps):
        raise RuntimeError(
            f"KITTI image/timestamp count mismatch in {sequence_dir}: "
            f"{len(images)} images versus {len(timestamps)} timestamps"
        )
    return [
        KittiFrame(index, timestamp, image)
        for index, (timestamp, image) in enumerate(zip(timestamps, images, strict=True))
    ]


class KittiPlayer(Node):
    """Play a contiguous KITTI odometry window with DPVO backpressure."""

    def __init__(self):
        super().__init__("kitti_player")
        self.declare_parameter("sequence_dir", "")
        self.declare_parameter("image_dir", "image_0")
        self.declare_parameter("calibration_key", "P0")
        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("camera_info_topic", "camera/camera_info")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("done_topic", "dpvo/player_done")
        self.declare_parameter("exit_on_finish", False)
        self.declare_parameter("stride", 2)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", 0)
        self.declare_parameter("max_frames", 0)

        sequence_dir = Path(self.get_parameter("sequence_dir").value)
        if not sequence_dir.is_dir():
            raise FileNotFoundError(
                f"KITTI sequence directory does not exist: {sequence_dir}"
            )
        image_dir = str(self.get_parameter("image_dir").value)
        self.frames = read_kitti_frames(sequence_dir, image_dir)
        calibration = read_kitti_calibration(sequence_dir / "calib.txt")
        calibration_key = str(self.get_parameter("calibration_key").value)
        if calibration_key not in calibration:
            raise KeyError(
                f"KITTI calibration {calibration_key!r} is not in "
                f"{sequence_dir / 'calib.txt'}"
            )
        projection = calibration[calibration_key]
        if projection.size != 12:
            raise ValueError(
                f"KITTI projection {calibration_key} must have 12 entries"
            )
        self.fx, self.fy, self.cx, self.cy = projection[[0, 5, 2, 6]]

        self.stride = int(self.get_parameter("stride").value)
        self.start_frame = int(self.get_parameter("start_frame").value)
        requested_end = int(self.get_parameter("end_frame").value)
        self.end_frame = requested_end if requested_end > 0 else len(self.frames)
        max_frames = int(self.get_parameter("max_frames").value)
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if not 0 <= self.start_frame < self.end_frame <= len(self.frames):
            raise ValueError(
                f"invalid KITTI window [{self.start_frame}, {self.end_frame}) "
                f"for {len(self.frames)} frames"
            )
        self.selected_frames = self.frames[
            self.start_frame : self.end_frame : self.stride
        ]
        if max_frames > 0:
            self.selected_frames = self.selected_frames[:max_frames]

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

        self.next_frame = 0
        self.waiting_for_ack = False
        self.finished = False
        self.shutdown_timer = None
        self.timer = self.create_timer(0.01, self._publish_next)
        self.get_logger().info(
            f"Playing KITTI {sequence_dir.name} raw window "
            f"[{self.start_frame}, {self.end_frame}) as "
            f"{len(self.selected_frames)} frames at stride {self.stride} "
            "with DPVO backpressure"
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
        image_data = cv2.imread(str(frame.image_path), cv2.IMREAD_GRAYSCALE)
        if image_data is None:
            raise RuntimeError(f"Failed to decode KITTI image: {frame.image_path}")
        image_data = np.ascontiguousarray(image_data)

        timestamp_ns = int(round(frame.timestamp * 1_000_000_000))
        image = Image()
        image.header.stamp.sec = timestamp_ns // 1_000_000_000
        image.header.stamp.nanosec = timestamp_ns % 1_000_000_000
        image.header.frame_id = "cam0"
        image.height, image.width = image_data.shape
        image.encoding = "mono8"
        image.is_bigendian = 0
        image.step = int(image.width)
        image.data = image_data.tobytes()

        info = CameraInfo()
        info.header = image.header
        info.height = image.height
        info.width = image.width
        info.k = [
            float(self.fx),
            0.0,
            float(self.cx),
            0.0,
            float(self.fy),
            float(self.cy),
            0.0,
            0.0,
            1.0,
        ]
        info.p = [
            float(self.fx),
            0.0,
            float(self.cx),
            0.0,
            0.0,
            float(self.fy),
            float(self.cy),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        info.d = []
        info.distortion_model = "plumb_bob"

        self.info_publisher.publish(info)
        self.image_publisher.publish(image)
        self.waiting_for_ack = True
        self.next_frame += 1
        if self.next_frame % 100 == 0:
            self.get_logger().info(
                f"Published and acknowledged {self.next_frame} frames; "
                f"source frame {frame.index}"
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
    node = KittiPlayer()
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

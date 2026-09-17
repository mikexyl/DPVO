"""Selected camera stream, rectified full-FOV and newest-frame only."""
import time

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


class OnlineCamera(Node):
    def __init__(self):
        super().__init__('dpvo_camera')
        for name, default in {'camera_type': 'realsense', 'camera_device': '/dev/dpvo_camera',
                              'camera_calibration': '', 'camera_serial': '', 'camera_fps': 8,
                              'camera_stride': 1, 'camera_format': 'yuyv',
                              'output_width': 384, 'output_height': 240}.items():
            self.declare_parameter(name, default)
        self.pipeline = None
        self.capture = None
        self.started = False
        self.previous = 0
        self.first_camera_time = None
        self.first_wall_ns = None
        self.format = self.get_parameter('camera_format').value
        if self.format not in ('bgr8', 'yuyv'):
            raise ValueError('camera_format must be bgr8 or yuyv')
        try:
            width = self.get_parameter('output_width').value
            height = self.get_parameter('output_height').value
            stride = self.get_parameter('camera_stride').value
            fps = self.get_parameter('camera_fps').value
            kind = self.get_parameter('camera_type').value
            if kind == 'zed':
                from deploy.jetson.zed_camera import LatestZedFrame, camera_maps
                self.maps, (fx, fy, cx, cy) = camera_maps(
                    self.get_parameter('camera_calibration').value, width, height)
                self.convert = lambda raw: raw
                self.capture = LatestZedFrame(self.get_parameter('camera_device').value, fps, stride)
            elif kind == 'realsense':
                import pyrealsense2 as rs
                from deploy.jetson.realsense_camera import LatestFrame, camera_maps, color_to_bgr
                self.pipeline = rs.pipeline()
                config = rs.config()
                serial = self.get_parameter('camera_serial').value
                if serial:
                    config.enable_device(serial)
                config.enable_stream(rs.stream.color, 1280, 800, getattr(rs.format, self.format), fps)
                profile = self.pipeline.start(config)
                self.started = True
                intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.maps, (fx, fy, cx, cy) = camera_maps(intr, width, height)
                self.convert = lambda raw: color_to_bgr(raw, self.format)
                self.capture = LatestFrame(self.pipeline, stride=stride)
            else:
                raise ValueError(f'Unknown camera_type: {kind}')
            self.info = CameraInfo()
            self.info.width, self.info.height = width, height
            self.info.distortion_model = 'plumb_bob'
            self.info.d = [0.] * 5
            self.info.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
            self.info.r = np.eye(3).ravel().tolist()
            self.info.p = [fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]
            image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            info_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.images = self.create_publisher(Image, 'camera/image_rect', image_qos)
            self.infos = self.create_publisher(CameraInfo, 'camera/camera_info', info_qos)
            self.infos.publish(self.info)
            self.create_timer(0.005, self.tick)
        except Exception:
            self.close()
            raise

    def tick(self):
        with self.capture.condition:
            if self.capture.error is not None:
                raise RuntimeError('Camera capture failed') from self.capture.error
            sample = self.capture.frame
        if sample is None or sample[0] == self.previous:
            return
        number, timestamp, _, raw = sample
        self.previous = number
        if self.first_camera_time is None:
            self.first_camera_time = timestamp
            self.first_wall_ns = time.time_ns()
        timestamp_ns = self.first_wall_ns + round((timestamp - self.first_camera_time) * 1e9)
        image = cv2.remap(self.convert(raw), *self.maps, cv2.INTER_LINEAR)
        message = Image()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(timestamp_ns, 10**9)
        message.header.frame_id = self.get_namespace().strip('/') + '/camera_optical'
        message.height, message.width = image.shape[:2]
        message.encoding, message.step = 'bgr8', message.width * 3
        message.data = image.tobytes()
        self.info.header = message.header
        self.infos.publish(self.info)
        self.images.publish(message)

    def close(self):
        if self.capture is not None:
            self.capture.stop.set()
            self.capture.thread.join(timeout=6)
            self.capture = None
        if self.started:
            self.pipeline.stop()
            self.started = False

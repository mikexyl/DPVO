"""Calibrated full-view sampling and bounded RealSense capture."""
import threading
import time

import numpy as np
import pyrealsense2 as rs


class LatestFrame:
    """A single replaceable frame slot; camera polling never waits for inference."""

    def __init__(self, pipeline, stride=1):
        if stride < 1:
            raise ValueError('Camera stride must be positive')
        self.pipeline = pipeline
        self.stride = stride
        self.condition = threading.Condition()
        self.frame = None
        self.count = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        try:
            while not self.stop.is_set():
                frames = self.pipeline.wait_for_frames(5000)
                color = frames.get_color_frame()
                if not color:
                    continue
                arrival = time.perf_counter()
                self.count += 1
                if (self.count - 1) % self.stride:
                    continue
                image = np.asanyarray(color.get_data()).copy()
                with self.condition:
                    self.frame = (self.count, color.get_timestamp() / 1000, arrival, image)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def get(self, previous):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error or (self.frame and self.frame[0] > previous), timeout=6)
            if self.error:
                raise RuntimeError("RealSense capture failed") from self.error
            if not ready:
                raise TimeoutError("No new RealSense frame for six seconds")
            return self.frame


def camera_maps(intrinsics, width, height):
    """Rectify and resize the entire camera view in one resampling operation."""
    if width * intrinsics.height != height * intrinsics.width:
        raise ValueError('Output must preserve the camera aspect ratio')
    scale = width / intrinsics.width
    fx, fy = intrinsics.fx * scale, intrinsics.fy * scale
    cx, cy = intrinsics.ppx * scale, intrinsics.ppy * scale
    # Ask librealsense itself to project each ideal pinhole ray into the source
    # image. This also handles its inverse/modified Brown conventions precisely.
    # Maps are built once, then reused by OpenCV for every incoming frame.
    mapped = np.array([
        rs.rs2_project_point_to_pixel(intrinsics, [
            (x - cx) / fx, (y - cy) / fy, 1.0])
        for y in range(height) for x in range(width)
    ], dtype=np.float32).reshape(height, width, 2)
    return (mapped[:, :, 0], mapped[:, :, 1]), [fx, fy, cx, cy]




def color_to_bgr(image, pixel_format):
    """Convert native YUYV without the SDK's CUDA-dependent color processor."""
    if pixel_format == 'yuyv':
        import cv2
        # SDK versions expose YUYV as either packed bytes or uint16 pixels.
        packed = image.view(np.uint8).reshape(image.shape[0], -1, 2)
        return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)
    if pixel_format == 'bgr8':
        return image
    raise ValueError(f'Unsupported camera color format: {pixel_format}')

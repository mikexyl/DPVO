"""ZED UVC left-camera capture without the ZED SDK or stereo depth engine."""
import configparser
import threading
import time

import cv2
import numpy as np


def camera_maps(calibration, width=384, height=240):
    """Undistort the complete VGA left view; scale both intrinsic axes explicitly."""
    config = configparser.ConfigParser()
    if not config.read(calibration):
        raise ValueError(f'Missing ZED factory calibration: {calibration}')
    section = config['LEFT_CAM_VGA']
    fx, fy, cx, cy = [section.getfloat(key) for key in ('fx', 'fy', 'cx', 'cy')]
    intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    distortion = np.array([section.getfloat(key) for key in ('k1', 'k2', 'p1', 'p2', 'k3')])
    if not np.isfinite(intrinsic).all() or not np.isfinite(distortion).all() or min(fx, fy, width, height) <= 0:
        raise ValueError('Invalid ZED calibration or output dimensions')
    output = intrinsic.copy()
    output[0] *= width / 672
    output[1] *= height / 376
    maps = cv2.initUndistortRectifyMap(intrinsic, distortion, np.eye(3), output,
                                     (width, height), cv2.CV_32FC1)
    return maps, [output[0, 0], output[1, 1], output[0, 2], output[1, 2]]


def left_image(frame):
    if frame.shape != (376, 1344, 3) or frame.dtype != np.uint8:
        raise ValueError(f'Expected ZED VGA stereo BGR image, got {frame.shape}, {frame.dtype}')
    return frame[:, :672].copy()


class LatestZedFrame:
    """Bounded newest-frame slot, matching the RealSense capture interface."""
    def __init__(self, device, fps=15, stride=2):
        if stride < 1 or fps <= 0:
            raise ValueError('Camera FPS and stride must be positive')
        self.condition = threading.Condition()
        self.frame = None
        self.error = None
        self.stop = threading.Event()
        self.video = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.video.isOpened():
            self.video.release()
            raise RuntimeError(f'Cannot open ZED video device {device}')
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
        self.video.set(cv2.CAP_PROP_FPS, fps)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.warmup_frames = round(fps * 2)
        self.stride = stride
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        count = 0
        try:
            while not self.stop.is_set():
                ok, frame = self.video.read()
                arrival = time.monotonic()
                if not ok:
                    raise RuntimeError('ZED UVC frame read failed')
                count += 1
                # UVC auto-exposure/white balance needs a short settling period.
                if count <= self.warmup_frames or (count - 1) % self.stride:
                    continue
                raw = left_image(frame)
                with self.condition:
                    self.frame = (count, arrival, arrival, raw)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.video.release()

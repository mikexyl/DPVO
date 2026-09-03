import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ros2.dpvo_multi_robot.dpvo_multi_robot.cu_multi_core import (
    build_pinhole_rectification,
    rectified_camera_info,
)
from ros2.dpvo_multi_robot.dpvo_multi_robot.newer_college_core import (
    PlaybackState,
    stamp_components,
)
from ros2.dpvo_multi_robot.dpvo_multi_robot.s3e_core import (
    decode_compressed_bgr8,
    read_s3e_calibration,
)


CALIBRATION = """%YAML:1.0
Camera.type: \"PinHole\"
Camera.fps: 10.0
LEFT.height: 4
LEFT.width: 6
LEFT.D: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [-0.05, 0.1, 0.0, 0.0, 0.0]
LEFT.K: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [100.0, 0.0, 2.5, 0.0, 101.0, 1.5, 0.0, 0.0, 1.0]
Tic: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0]
"""


class S3EPlayerCoreTest(unittest.TestCase):
    def test_calibration_decode_rectification_and_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha.yaml"
            path.write_text(CALIBRATION)
            calibration = read_s3e_calibration(path)
            self.assertEqual(calibration.camera.width, 6)
            self.assertEqual(calibration.camera.height, 4)
            self.assertEqual(calibration.fps, 10.0)
            np.testing.assert_allclose(calibration.t_imu_camera[:3, 3], [0.1, 0.2, 0.3])

            source = np.zeros((4, 6, 3), dtype=np.uint8)
            source[:, :, 1] = 127
            success, encoded = cv2.imencode(".jpg", source)
            self.assertTrue(success)
            decoded = decode_compressed_bgr8(encoded.tobytes())
            self.assertEqual(decoded.shape, (4, 6, 3))
            self.assertTrue(decoded.flags.c_contiguous)

            rectification = build_pinhole_rectification(calibration.camera)
            rectified = rectification.rectify(decoded)
            self.assertEqual(rectified.shape, (4, 6, 3))
            info = rectified_camera_info(rectification)
            self.assertEqual(info["width"], 6)
            self.assertEqual(info["height"], 4)
            self.assertEqual(info["distortion_model"], "plumb_bob")
            self.assertEqual(info["d"], [])
            self.assertEqual(info["k"][0], rectification.camera_matrix[0, 0])

    def test_timestamp_stride_backpressure_and_completion(self):
        stamp = SimpleNamespace(sec=1_661_159_289, nanosec=835_962_500)
        self.assertEqual(stamp_components(stamp), (1_661_159_289, 835_962_500))
        state = PlaybackState(stride=2, start_frame=1, max_frames=2)
        selected = []
        for _ in range(10):
            if state.select_next_source():
                selected.append(state.source_index)
                state.mark_published()
                with self.assertRaisesRegex(RuntimeError, "before acknowledgement"):
                    state.select_next_source()
                if state.acknowledge():
                    break
        self.assertEqual(selected, [2, 4])
        self.assertTrue(state.limit_reached)
        state.finish()
        self.assertTrue(state.finished)

    def test_invalid_compressed_payload_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_compressed_bgr8(b"")
        with self.assertRaisesRegex(ValueError, "failed"):
            decode_compressed_bgr8(b"not a jpeg")


if __name__ == "__main__":
    unittest.main()

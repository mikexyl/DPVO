import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ros2.dpvo_multi_robot.dpvo_multi_robot.newer_college_core import (
    PlaybackState,
    build_fisheye_rectification,
    decode_compressed_mono8,
    read_kalibr_camera_calibration,
    rectified_camera_info,
    stamp_components,
)


CALIBRATION = """cam0:
  camera_model: pinhole
  intrinsics: [330.0, 331.0, 360.0, 270.0]
  distortion_model: equidistant
  distortion_coeffs: [-0.02, 0.001, 0.0, 0.0]
  resolution: [720, 540]
  T_cam_imu:
    - [1.0, 0.0, 0.0, 0.01]
    - [0.0, 1.0, 0.0, 0.02]
    - [0.0, 0.0, 1.0, 0.03]
    - [0.0, 0.0, 0.0, 1.0]
"""


class NewerCollegePlayerCoreTest(unittest.TestCase):
    def _calibration(self, root: Path):
        path = root / "camchain.yaml"
        path.write_text(CALIBRATION)
        return read_kalibr_camera_calibration(path)

    def test_decode_rectify_and_updated_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = self._calibration(Path(directory))
            source = np.tile(np.arange(720, dtype=np.uint8), (540, 1))
            success, encoded = cv2.imencode(".jpg", source)
            self.assertTrue(success)
            decoded = decode_compressed_mono8(encoded.tobytes())
            self.assertEqual(decoded.shape, (540, 720))
            self.assertEqual(decoded.dtype, np.uint8)

            rectification = build_fisheye_rectification(calibration, balance=0.0)
            rectified = rectification.rectify(decoded)
            self.assertEqual(rectified.shape, (540, 720))
            self.assertTrue(rectified.flags.c_contiguous)
            info = rectified_camera_info(rectification)
            self.assertEqual(info["width"], 720)
            self.assertEqual(info["height"], 540)
            self.assertEqual(info["d"], [])
            self.assertEqual(info["distortion_model"], "plumb_bob")
            self.assertEqual(info["k"][0], rectification.camera_matrix[0, 0])
            self.assertEqual(info["k"][4], rectification.camera_matrix[1, 1])
            self.assertNotEqual(info["k"][0], calibration.camera_matrix[0, 0])

    def test_timestamp_uses_header_components(self):
        stamp = SimpleNamespace(sec=1_624_000_123, nanosec=456_789_012)
        self.assertEqual(stamp_components(stamp), (1_624_000_123, 456_789_012))
        ros1_stamp = SimpleNamespace(sec=9, nsec=12)
        self.assertEqual(stamp_components(ros1_stamp), (9, 12))

    def test_stride_backpressure_limit_and_completion(self):
        state = PlaybackState(stride=2, start_frame=1, max_frames=2)
        selected_sources = []
        for _ in range(20):
            if state.select_next_source():
                selected_sources.append(state.source_index)
                state.mark_published()
                with self.assertRaisesRegex(RuntimeError, "before acknowledgement"):
                    state.select_next_source()
                reached_limit = state.acknowledge()
                if reached_limit:
                    break
        self.assertEqual(selected_sources, [2, 4])
        self.assertEqual(state.processed, 2)
        self.assertTrue(state.limit_reached)
        state.finish()
        self.assertTrue(state.finished)

    def test_completion_requires_final_ack(self):
        state = PlaybackState(stride=1, start_frame=0, max_frames=1)
        self.assertTrue(state.select_next_source())
        state.mark_published()
        with self.assertRaisesRegex(RuntimeError, "unacknowledged"):
            state.finish()
        self.assertTrue(state.acknowledge())
        state.finish()


if __name__ == "__main__":
    unittest.main()

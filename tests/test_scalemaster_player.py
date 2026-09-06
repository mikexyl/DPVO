import tempfile
import unittest
from pathlib import Path

import numpy as np

from ros2.dpvo_multi_robot.dpvo_multi_robot.scalemaster_core import (
    read_camera_matrix,
    read_frames,
    read_odometry,
)


class ScaleMasterPlayerCoreTest(unittest.TestCase):
    def _sequence(self, root: Path) -> Path:
        sequence = root / "Library_01"
        frames = sequence / "frames"
        frames.mkdir(parents=True)
        (sequence / "camera_matrix.csv").write_text(
            "100.0, 0.0, 2.0\n0.0, 101.0, 1.0\n0.0, 0.0, 1.0",
            encoding="utf-8",
        )
        (sequence / "odometry.csv").write_text(
            "timestamp, frame, x, y, z, qx, qy, qz, qw\n"
            "10.0,000000,0,0,0,0,0,0,2\n"
            "10.1,000001,1,0,0,0,0,0,1\n"
            "10.2,000002,2,0,0,0,0,0,1\n",
            encoding="utf-8",
        )
        # Real ScaleMaster archives can contain one odometry sample without a
        # corresponding extracted RGB frame. Frame-ID joining must ignore it.
        (frames / "frame_00000.jpg").write_bytes(b"not decoded by core")
        (frames / "frame_00001.jpg").write_bytes(b"not decoded by core")
        return sequence

    def test_calibration_odometry_and_frame_id_join(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = self._sequence(Path(directory))
            matrix = read_camera_matrix(sequence / "camera_matrix.csv")
            self.assertEqual(matrix.shape, (3, 3))
            self.assertEqual(matrix[1, 1], 101.0)
            poses = read_odometry(sequence / "odometry.csv")
            self.assertEqual(len(poses), 3)
            np.testing.assert_allclose(poses[0].quaternion_xyzw, [0, 0, 0, 1])
            frames = read_frames(sequence)
            self.assertEqual([frame.frame_id for frame in frames], [0, 1])
            self.assertEqual([frame.timestamp for frame in frames], [10.0, 10.1])

    def test_missing_odometry_and_duplicate_image_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = self._sequence(Path(directory))
            (sequence / "frames" / "frame_00003.jpg").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "has no row"):
                read_frames(sequence)
            (sequence / "frames" / "frame_00003.jpg").unlink()
            (sequence / "frames" / "frame_00000.png").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_frames(sequence)

    def test_non_increasing_timestamps_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = self._sequence(Path(directory))
            path = sequence / "odometry.csv"
            path.write_text(
                "timestamp,frame,x,y,z,qx,qy,qz,qw\n"
                "10.0,0,0,0,0,0,0,0,1\n"
                "9.0,1,0,0,0,0,0,0,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-increasing"):
                read_odometry(path)


if __name__ == "__main__":
    unittest.main()

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from deploy.blackwell_ros2 import graco_staged
from deploy.blackwell_ros2.export_graco_evo_tum import (
    read_imu_groundtruth,
    read_t_imu_cam0,
)


class GracoStagingTest(unittest.TestCase):
    def test_groundtruth_pose_is_transformed_from_imu_to_cam0(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "stereo-imu.yaml"
            calibration.write_text(
                "T_Imu_cam0:\n"
                "  rows: 4\n"
                "  cols: 4\n"
                "  data: [1, 0, 0, 0.2, 0, 1, 0, -0.1, "
                "0, 0, 1, 0.05, 0, 0, 0, 1]\n",
                encoding="utf-8",
            )
            groundtruth = root / "aerial.txt"
            groundtruth.write_text(
                "10.0 1 2 3 0 0 0 1\n10.005 2 3 4 0 0 0 2\n",
                encoding="utf-8",
            )
            transform = read_t_imu_cam0(calibration)
            rows = read_imu_groundtruth(groundtruth, transform)
        np.testing.assert_allclose(rows[0][1], [1.2, 1.9, 3.05])
        np.testing.assert_allclose(rows[0][2], [0, 0, 0, 1])
        self.assertAlmostEqual(float(np.linalg.norm(rows[1][2])), 1.0)

    def test_preflight_gate_records_in_range_keyframe_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "robot0"
            artifact.mkdir()
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "input_frame_count": 300,
                        "keyframe_count": 120,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "gate.json"
            args = argparse.Namespace(
                artifact=artifact,
                source_rate_hz=20.0,
                stride=2,
                minimum=3.0,
                maximum=5.0,
                output=output,
            )
            graco_staged.check_preflight_rate(args)
            result = json.loads(output.read_text())
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["keyframe_rate_hz"], 120 / 29.9)

    def test_input_rate_gate_separates_feed_from_retained_keyframes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "robot0"
            artifact.mkdir()
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "input_frame_count": 300,
                        "keyframe_count": 17,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "gate.json"
            args = argparse.Namespace(
                artifact=artifact,
                source_rate_hz=20.0,
                stride=4,
                minimum=3.0,
                maximum=5.0,
                output=output,
            )
            graco_staged.check_preflight_input_rate(args)
            result = json.loads(output.read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(result["input_rate_hz"], 5.0)
        self.assertLess(result["retained_keyframe_rate_hz"], 1.0)

    def test_manifest_records_all_four_sequences_and_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            common = {}
            for name in (
                "vocabulary",
                "network",
                "config",
                "stereo_calibration",
                "stereo_imu_calibration",
                "groundtruth_calibration",
            ):
                path = root / name
                path.write_bytes(name.encode())
                common[name] = path
            source_bags, bags, groundtruth = [], [], []
            for robot in ("robot0", "robot1", "robot2", "robot3"):
                source = root / f"{robot}.bag"
                source.write_bytes(f"source-{robot}".encode())
                bag = root / f"{robot}-ros2"
                bag.mkdir()
                (bag / "metadata.yaml").write_text("version: 8\n")
                (bag / f"{robot}.db3").write_bytes(f"db-{robot}".encode())
                truth = root / f"{robot}.txt"
                truth.write_text("1 0 0 0 0 0 0 1\n")
                source_bags.append(f"{robot}={source}")
                bags.append(f"{robot}={bag}")
                groundtruth.append(f"{robot}={truth}")
            args = argparse.Namespace(
                run_dir=run,
                source_bag=source_bags,
                bag=bags,
                groundtruth=groundtruth,
                vocabulary=common["vocabulary"],
                network=common["network"],
                config=common["config"],
                stereo_calibration=common["stereo_calibration"],
                stereo_imu_calibration=common["stereo_imu_calibration"],
                groundtruth_calibration=common["groundtruth_calibration"],
                parameter=["tracking.stride=4", "runtime.cuda_mps=true"],
            )
            graco_staged.record_manifest(args)
            manifest = json.loads((run / "experiment_manifest.json").read_text())
        self.assertEqual(manifest["format"], graco_staged.FORMAT)
        self.assertEqual(
            manifest["scenarios"]["four"]["robot_ids"],
            ["robot0", "robot1", "robot2", "robot3"],
        )
        self.assertIs(manifest["parameters"]["runtime.cuda_mps"], True)
        self.assertIn("stereo_imu_calibration", manifest)


if __name__ == "__main__":
    unittest.main()

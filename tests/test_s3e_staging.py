import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deploy.blackwell_ros2.export_s3e_evo_tum import read_position_groundtruth
from deploy.blackwell_ros2.plot_joint_map_ply import associate
from deploy.blackwell_ros2.s3e_staged import record_manifest


class S3EStagingTest(unittest.TestCase):
    def test_position_groundtruth_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha_gt.txt"
            path.write_text(
                "10.0 744548.0 2553471.0 -0.2 0 0 0 1\n"
                "11.0 744549.0 2553472.0 -0.1 0 0 0 2\n"
            )
            rows = read_position_groundtruth(path)
            self.assertEqual(len(rows), 2)
            np.testing.assert_allclose(rows[0][1], [744548.0, 2553471.0, -0.2])
            np.testing.assert_allclose(rows[1][2], [0.0, 0.0, 0.0, 1.0])

    def test_non_increasing_groundtruth_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text(
                "10.0 0 0 0 0 0 0 1\n"
                "10.0 1 1 1 0 0 0 1\n"
            )
            with self.assertRaisesRegex(ValueError, "non-increasing"):
                read_position_groundtruth(path)

    def test_sparse_reference_association_matches_each_reference_once(self):
        reference = np.array([0.0, 1.0, 2.0])
        estimate = np.arange(-0.1, 2.11, 0.1)
        reference_indices, estimate_indices = associate(
            reference, estimate, max_difference=0.11
        )
        np.testing.assert_array_equal(reference_indices, [0, 1, 2])
        np.testing.assert_allclose(estimate[estimate_indices], reference, atol=1e-12)

    def test_manifest_records_selected_sequence_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "S3E_Playground_2"
            bag.mkdir()
            (bag / "metadata.yaml").write_text("metadata")
            (bag / "S3E_Playground_2.db3").write_bytes(b"bag")
            shared = {}
            for name in ("vocab", "network", "config"):
                shared[name] = root / name
                shared[name].write_text(name)
            calibrations = []
            groundtruth = []
            topics = []
            for index in range(3):
                robot = f"robot{index}"
                calibration = root / f"{robot}.yaml"
                gt = root / f"{robot}_gt.txt"
                calibration.write_text("calibration")
                gt.write_text("groundtruth")
                calibrations.append(f"{robot}={calibration}")
                groundtruth.append(f"{robot}={gt}")
                topics.append(f"{robot}=/{robot}/left_camera/compressed")
            run_dir = root / "run"
            record_manifest(
                SimpleNamespace(
                    run_dir=run_dir,
                    bag=bag,
                    dataset_label="S3Ev1 Playground 2",
                    version_label="s3e_playground_2_staged_v1",
                    calibration=calibrations,
                    groundtruth=groundtruth,
                    topic=topics,
                    vocabulary=shared["vocab"],
                    network=shared["network"],
                    config=shared["config"],
                    parameter=["tracking.stride=2"],
                )
            )
            manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
            self.assertEqual(manifest["dataset"], "S3Ev1 Playground 2")
            self.assertEqual(
                manifest["material_passport"]["version_label"],
                "s3e_playground_2_staged_v1",
            )


if __name__ == "__main__":
    unittest.main()

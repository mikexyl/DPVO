import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from deploy.blackwell_ros2.analyze_cu_multi_tracking import align_similarity
from deploy.blackwell_ros2 import cu_multi_staged
from deploy.blackwell_ros2.export_cu_multi_evo_tum import read_utm_groundtruth


class CuMultiStagingTest(unittest.TestCase):
    def test_similarity_alignment_recovers_known_transform(self):
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = 2.5 * source @ rotation.T + np.asarray([5.0, -3.0, 7.0])
        aligned, scale, recovered_rotation, translation = align_similarity(source, target)
        np.testing.assert_allclose(aligned, target, atol=1e-12)
        self.assertAlmostEqual(scale, 2.5)
        np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(translation, [5.0, -3.0, 7.0], atol=1e-12)

    def test_utm_reader_skips_header_and_normalizes_quaternions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gt.csv"
            path.write_text(
                "timestamp,x,y,z,qx,qy,qz,qw\n"
                "10.0,500000,4400000,1600,0,0,0,2\n"
                "10.05,500001,4400001,1601,0,0,1,1\n",
                encoding="utf-8",
            )
            rows = read_utm_groundtruth(path)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0][2][3]), 1.0)
        self.assertAlmostEqual(float((rows[1][2] ** 2).sum()), 1.0)

    def test_utm_reader_rejects_non_increasing_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gt.csv"
            path.write_text(
                "10,0,0,0,0,0,0,1\n10,1,0,0,0,0,0,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-increasing"):
                read_utm_groundtruth(path)

    def test_manifest_records_archives_extracted_bags_and_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            common = {}
            for name in ("vocab", "network", "config"):
                path = root / name
                path.write_bytes(name.encode())
                common[name] = path
            archives, bags, truths = [], [], []
            for robot in ("robot0", "robot1"):
                archive = root / f"{robot}.zip"
                archive.write_bytes(f"archive-{robot}".encode())
                bag = root / f"{robot}-bag"
                bag.mkdir()
                (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
                (bag / f"{robot}.db3").write_bytes(f"db-{robot}".encode())
                truth = root / f"{robot}.csv"
                truth.write_text("1,0,0,0,0,0,0,1\n")
                archives.append(f"{robot}={archive}")
                bags.append(f"{robot}={bag}")
                truths.append(f"{robot}={truth}")
            args = argparse.Namespace(
                run_dir=run,
                scenario="two",
                source_archive=archives,
                bag=bags,
                groundtruth=truths,
                vocabulary=common["vocab"],
                network=common["network"],
                config=common["config"],
                parameter=["tracking.stride=2", "evaluation.align=true"],
            )
            cu_multi_staged.record_manifest(args)
            manifest = json.loads((run / "experiment_manifest.json").read_text())
        self.assertEqual(manifest["format"], cu_multi_staged.FORMAT)
        self.assertEqual(manifest["scenarios"]["two"]["robot_ids"], ["robot0", "robot1"])
        self.assertEqual(manifest["parameters"]["tracking.stride"], 2)
        self.assertIs(manifest["parameters"]["evaluation.align"], True)
        self.assertIn("database", manifest["tracking"]["sources"]["robot0"]["extracted_ros2_bag"])


if __name__ == "__main__":
    unittest.main()

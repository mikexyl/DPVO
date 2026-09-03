import importlib.util
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import zipfile

import numpy as np

from dpvo.loop_closure.centralized import Sim3
from dpvo.loop_closure.pose_graph import (
    PoseGraphEdge,
    PoseGraphVertex,
    Sim3PoseGraph,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


staged = load_script(
    "newer_college_staged_test_module",
    "deploy/blackwell_ros2/newer_college_staged.py",
)
exporter = load_script(
    "newer_college_export_test_module",
    "deploy/blackwell_ros2/export_newer_college_evo_tum.py",
)


class NewerCollegeStagingTest(unittest.TestCase):
    def test_manifest_records_inputs_and_pins_tracking_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for name in (
                "bag",
                "groundtruth",
                "calibration",
                "vocabulary",
                "network",
                "config",
            ):
                path = root / name
                path.write_bytes(name.encode("utf-8"))
                inputs[name] = path
            arguments = Namespace(
                run_dir=root / "run",
                scenario="two",
                calibration=inputs["calibration"],
                vocabulary=inputs["vocabulary"],
                network=inputs["network"],
                config=inputs["config"],
                source=[f"robot0={inputs['bag']}"],
                groundtruth=[f"robot0={inputs['groundtruth']}"],
                inventory=[],
                parameter=["tracking.stride=2", "stage3.random_seed=42"],
            )
            staged.record_manifest(arguments)
            manifest_path = arguments.run_dir / "experiment_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["parameters"]["tracking.stride"], 2)
            self.assertEqual(
                manifest["tracking"]["sources"]["robot0"]["bag"]["sha256"],
                staged.sha256(inputs["bag"]),
            )

            artifact = arguments.run_dir / "tracking" / "robot0" / "manifest.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")
            arguments.parameter = ["tracking.stride=3"]
            with self.assertRaisesRegex(RuntimeError, "tracking.stride"):
                staged.record_manifest(arguments)

    def test_base_groundtruth_is_transformed_to_cam0(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ground_truth.csv"
            path.write_text(
                "sec,nsec,x,y,z,qx,qy,qz,qw\n"
                "100,250000000,1,2,3,0,0,0,1\n",
                encoding="utf-8",
            )
            t_base_cam = np.eye(4)
            t_base_cam[:3, 3] = [0.1, -0.2, 0.3]
            rows = exporter.read_base_groundtruth(path, t_base_cam)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0][0], 100.25)
            np.testing.assert_allclose(rows[0][1], [1.1, 1.8, 3.3])
            np.testing.assert_allclose(rows[0][2], [0.0, 0.0, 0.0, 1.0])

    def test_graph_gate_counts_pairs_and_connectivity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            vertices = [
                PoseGraphVertex(
                    vertex_id=index,
                    robot_id=f"robot{index}",
                    session_id=f"robot{index}_tracking",
                    keyframe_id=0,
                    estimate=Sim3.identity(),
                    fixed=index == 0,
                )
                for index in range(3)
            ]
            edges = [
                PoseGraphEdge(
                    edge_id=index,
                    source=source,
                    target=target,
                    measurement=Sim3.identity(),
                    information_diagonal=np.ones(7),
                    edge_type="inter_robot_loop_closure",
                )
                for index, (source, target) in enumerate(
                    [(0, 1), (0, 1), (1, 2)]
                )
            ]
            write_json(
                Sim3PoseGraph(
                    name="verified",
                    vertices=vertices,
                    edges=edges,
                    metadata={
                        "pipeline_stage": "geometric_verification",
                        "input_contains_global_optimization": False,
                    },
                ),
                path,
            )
            diagnostics = staged.graph_diagnostics(
                path, ["robot0", "robot1", "robot2"]
            )
            self.assertTrue(diagnostics["connected"])
            self.assertEqual(diagnostics["inter_robot_loop_count"], 3)
            self.assertEqual(diagnostics["pair_loop_counts"]["robot0--robot1"], 2)

    def test_evo_archive_reports_positive_sim3_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.zip"
            transform = np.eye(4)
            transform[:3, :3] *= 2.5
            array = io.BytesIO()
            np.save(array, transform)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("stats.json", json.dumps({"rmse": 0.4}))
                archive.writestr(
                    "alignment_transformation_sim3.npy", array.getvalue()
                )
            result = staged.evo_result(path)
            self.assertAlmostEqual(result["ate_rmse_m"], 0.4)
            self.assertAlmostEqual(result["groundtruth_relative_scale"], 2.5)


if __name__ == "__main__":
    unittest.main()

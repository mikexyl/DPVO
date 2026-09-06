"""Regression checks for the existing paper renderer's stage references."""

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stage_trajectory_renderer",
    REPO_ROOT / "paper/icra2027/scripts/plot_iphone_cbs_teaser.py",
)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


class TestStageGroundTruth(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "groundtruth_robot0.tum"
        self.path.write_text(
            "0 1 2 3 0 0 0 1\n1 4 5 6 0 0 0 1\n2 7 8 9 0 0 0 1\n"
        )
        self.graph = {
            "vertices": [
                {"robot_id": "robot0", "keyframe_id": 1, "timestamp": 2.001},
                {"robot_id": "robot0", "keyframe_id": 0, "timestamp": 0.001},
            ]
        }

    def load(self):
        return RENDERER.load_stage_groundtruth(
            REPO_ROOT, self.root, self.graph, ("robot0",)
        )

    def test_association_uses_timestamp_and_keyframe_order(self):
        reference, paths = self.load()
        np.testing.assert_array_equal(reference["robot0"], [[1, 2, 3], [7, 8, 9]])
        self.assertEqual(paths["groundtruth_robot0"], self.path)
        self.assertEqual(paths["reference_loader"].name, "plot_kitti_trajectories.py")

    def test_missing_association_is_not_silently_dropped(self):
        self.graph["vertices"][0]["timestamp"] = 3.0
        with self.assertRaisesRegex(ValueError, "not every robot0"):
            self.load()

    def test_duplicate_association_is_rejected(self):
        self.graph["vertices"][0]["timestamp"] = 0.002
        with self.assertRaisesRegex(ValueError, "duplicate ground-truth"):
            self.load()

    def test_unsorted_ground_truth_is_rejected(self):
        self.path.write_text("1 4 5 6 0 0 0 1\n0 1 2 3 0 0 0 1\n")
        with self.assertRaisesRegex(ValueError, "timestamps must increase"):
            self.load()


class TestStageLoopSegments(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "snapshot.csv"
        # Noncontiguous IDs, reversed row order, and a nonzero Y coordinate.
        self.path.write_text(
            "vertex_id,tx,ty,tz\n42,8,77,9\n7,2,66,3\n"
        )
        self.graph = {"edges": [
            {"id": 100, "type": "visual_odometry", "source": 7, "target": 42},
            {"id": 101, "type": "inter_robot_loop_closure", "source": 7, "target": 42},
        ]}

    def test_exact_vertex_lookup_and_only_verified_loops(self):
        segments, edges = RENDERER.stage_loop_segments(self.graph, self.path)
        np.testing.assert_array_equal(segments, [[[2, 66, 3], [8, 77, 9]]])
        self.assertEqual(edges, [{"edge_id": 101, "source": 7, "target": 42}])

    def test_missing_endpoint_is_not_silently_dropped(self):
        self.graph["edges"][1]["target"] = 999
        with self.assertRaisesRegex(ValueError, "endpoint missing"):
            RENDERER.stage_loop_segments(self.graph, self.path)

    def test_duplicate_vertex_is_rejected(self):
        self.path.write_text("vertex_id,tx,ty,tz\n7,1,2,3\n7,4,5,6\n")
        with self.assertRaisesRegex(ValueError, "duplicate snapshot vertex"):
            RENDERER.stage_loop_segments(self.graph, self.path)

    def test_nonfinite_position_is_rejected(self):
        self.path.write_text("vertex_id,tx,ty,tz\n7,nan,2,3\n")
        with self.assertRaisesRegex(ValueError, "nonfinite snapshot vertex"):
            RENDERER.stage_loop_segments(self.graph, self.path)

    def test_duplicate_loop_id_is_rejected(self):
        self.graph["edges"].append(dict(self.graph["edges"][1]))
        with self.assertRaisesRegex(ValueError, "duplicate loop edge"):
            RENDERER.stage_loop_segments(self.graph, self.path)

    def test_empty_loop_set_keeps_segment_dimensions(self):
        self.graph["edges"] = self.graph["edges"][:1]
        segments, edges = RENDERER.stage_loop_segments(self.graph, self.path)
        self.assertEqual(segments.shape, (0, 2, 3))
        self.assertEqual(edges, [])


class TestGroundTruthDisplayTransform(unittest.TestCase):
    def test_inverts_joint_fit_including_rotation_translation_and_scale(self):
        rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
        translation = np.array([10., -20., 30.])
        scale = 4.0
        reference = {
            "robot0": np.array([[8., 2., 1.], [1., 3., -2.]]),
            "robot1": np.array([[40., 7., 4.], [9., -8., 20.]]),
        }
        original = {robot: points.copy() for robot, points in reference.items()}
        displayed, inverse = RENDERER.groundtruth_in_solver_frame(reference, {
            "scale": scale, "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        })
        self.assertEqual(inverse["scale"], 0.25)
        for robot in reference:
            np.testing.assert_allclose(
                RENDERER.apply_sim3(displayed[robot], (scale, rotation, translation)),
                reference[robot], atol=1e-12,
            )
            np.testing.assert_array_equal(reference[robot], original[robot])

    def test_invalid_scale_is_rejected(self):
        for scale in (0., -1., float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "joint scale"):
                RENDERER.groundtruth_in_solver_frame({}, {
                    "scale": scale, "rotation": np.eye(3), "translation": np.zeros(3),
                })


class TestSingleStageSnapshot(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dpgo = self.root / "dpgo"
        self.dpgo.mkdir()
        self.points = {
            "robot0": np.array([[0, 0, 0], [1, 0, 2], [2, 1, 1], [3, 0, 4.]]),
            "robot1": np.array([[4, 0, 1], [3, 1, 2], [2, 0, 4], [5, 2, 5.]]),
        }
        self.reference = self.root / "common_reference.csv"
        vertices = []
        snapshot_rows = []
        reference_rows = []
        for robot, points in self.points.items():
            for keyframe, point in enumerate(points):
                vertex = len(vertices)
                vertices.append({"id": vertex, "robot_id": robot, "keyframe_id": keyframe})
                snapshot_rows.append([vertex, robot, keyframe, *point])
                reference_rows.append([vertex, robot, keyframe, *(3 * point + 30)])
        for path, data in (
            (self.dpgo / "cbs_iteration_52.csv", snapshot_rows),
            (self.reference, reference_rows),
        ):
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["vertex_id", "robot_id", "keyframe_id", "tx", "ty", "tz"])
                writer.writerows(data)
        (self.dpgo / "input_keyframes_unoptimized.json").write_text(
            json.dumps({"vertices": vertices, "edges": [
                {"id": 100, "type": "inter_robot_loop_closure", "source": 1, "target": 6},
            ]})
        )
        (self.dpgo / "cbs_trajectory_snapshots.csv").write_text(
            "iteration,alignment_frame\n52,robot0_anchor_estimates\n"
        )
        self.provenance = {
            "status": "complete", "return_code": 0,
            "command": [
                "cbs", "--iterations=100", "--stage_mode=fixed",
                "--anchor_start_iteration=2", "--pose_warmup_iterations=0",
                "--target_hellinger=0.1", "--d_reset=1.1", "--random_seed=42",
                "--bootstrap_robot_anchors=false",
                "--trajectory_snapshot_iterations=0,2,52,100",
            ],
        }
        self.save_provenance()

    def save_provenance(self):
        (self.dpgo / "offline_dpgo_provenance.json").write_text(json.dumps(self.provenance))

    def render(self, show_loop_edges=False):
        RENDERER.render_fourteen_robot_stage_trajectories(
            REPO_ROOT, self.root / "plots", "test", self.root,
            expected_robot_count=2, selected_iteration=52,
            reference_csv=self.reference,
            show_loop_edges=show_loop_edges,
        )

    def test_selected_snapshot_uses_native_coordinates_and_actual_schedule(self):
        from matplotlib.axes import Axes

        captured = []
        original_plot = Axes.plot

        def capture(axis, x, y, **kwargs):
            captured.append(np.column_stack((x, y)))
            return original_plot(axis, x, y, **kwargs)

        with mock.patch.object(Axes, "plot", new=capture):
            self.render()
        for index, points in enumerate(self.points.values()):
            for rendered in captured[2 * index:2 * index + 2]:
                np.testing.assert_array_equal(rendered, points[:, [0, 2]])
        self.assertEqual(len(captured), 4)
        trace = json.loads((self.root / "plots/test_trace.json").read_text())
        self.assertEqual(trace["iterations"], [52])
        self.assertEqual(trace["round_counts_at_snapshot"], {"pose": 2, "anchor": 50})
        self.assertEqual(trace["stage_boundaries"]["anchor_stage"], [3, 100])
        self.assertEqual(trace["presentation"]["trajectory_alignment"], "none")
        self.assertFalse(trace["presentation"]["labels"])
        self.assertNotIn("loop_overlay", trace)
        self.assertFalse(trace["ate_evaluation"]["reference_is_ground_truth"])
        self.assertLess(trace["ate_evaluation"]["metrics"]["52"]["joint_ate"], 1e-12)
        self.assertTrue((self.root / "plots/test_iteration_52.pdf").is_file())

    def test_loop_overlay_uses_snapshot_coordinates_not_reference(self):
        from matplotlib.axes import Axes
        from matplotlib.collections import LineCollection

        captured = []
        original_add = Axes.add_collection

        def capture(axis, collection, **kwargs):
            if isinstance(collection, LineCollection):
                captured.extend(collection.get_segments())
            return original_add(axis, collection, **kwargs)

        with mock.patch.object(Axes, "add_collection", new=capture):
            self.render(show_loop_edges=True)
        np.testing.assert_array_equal(captured, [[[1, 2], [2, 4]]])
        trace = json.loads((self.root / "plots/test_trace.json").read_text())
        self.assertEqual(trace["loop_overlay"]["edge_count"], 1)
        self.assertEqual(
            trace["loop_overlay"]["segments_xyz"]["52"],
            [[[1, 0, 2], [2, 0, 4]]],
        )
        self.assertEqual(trace["presentation"]["trajectory_alignment"], "none")
        self.assertEqual(trace["round_counts_at_snapshot"], {"pose": 2, "anchor": 50})
        self.assertLess(trace["ate_evaluation"]["metrics"]["52"]["joint_ate"], 1e-12)

    def test_unrequested_snapshot_is_rejected(self):
        self.provenance["command"][-1] = "--trajectory_snapshot_iterations=0,2,100"
        self.save_provenance()
        with self.assertRaisesRegex(ValueError, "not requested"):
            self.render()

    def test_failed_replay_is_rejected(self):
        self.provenance.update(status="failed", return_code=1)
        self.save_provenance()
        with self.assertRaisesRegex(ValueError, "successfully completed"):
            self.render()

    def test_centralized_bootstrap_does_not_relabel_cbs_initialization(self):
        self.provenance["command"] = [
            option.replace("--bootstrap_robot_anchors=false", "--bootstrap_robot_anchors=true")
            for option in self.provenance["command"]
        ]
        self.save_provenance()
        self.render()
        trace = json.loads((self.root / "plots/test_trace.json").read_text())
        self.assertEqual(trace["solver_options"]["bootstrap_robot_anchors"], "true")
        self.assertEqual(trace["round_counts_at_snapshot"], {"pose": 2, "anchor": 50})

    def test_groundtruth_overlay_preserves_estimates_and_uses_one_joint_fit(self):
        from matplotlib.axes import Axes
        from matplotlib.collections import LineCollection

        reference = {robot: 3 * points + 30 for robot, points in self.points.items()}
        # Genuine inter-robot misalignment must not be hidden by separate fits.
        reference["robot1"] = reference["robot1"] + [10, -5, 20]
        robots = tuple(self.points)
        _, metrics = RENDERER.centralized_reference_ate(self.points, reference, robots)
        expected, _ = RENDERER.groundtruth_in_solver_frame(reference, metrics["joint_alignment"])
        captured, segments = [], []
        original_plot, original_add = Axes.plot, Axes.add_collection

        def capture(axis, x, y, **kwargs):
            captured.append(np.column_stack((x, y)))
            return original_plot(axis, x, y, **kwargs)

        def capture_segments(axis, collection, **kwargs):
            if isinstance(collection, LineCollection):
                segments.extend(collection.get_segments())
            return original_add(axis, collection, **kwargs)

        with (
            mock.patch.object(Axes, "plot", new=capture),
            mock.patch.object(Axes, "add_collection", new=capture_segments),
            mock.patch.object(RENDERER, "load_stage_groundtruth", return_value=(reference, {})),
        ):
            RENDERER.render_fourteen_robot_stage_trajectories(
                REPO_ROOT, self.root / "plots", "test", self.root,
                expected_robot_count=2, selected_iteration=52,
                groundtruth_dir=self.root, show_groundtruth=True, show_loop_edges=True,
            )
        self.assertEqual(len(captured), 6)
        for index, robot in enumerate(robots):
            for rendered in captured[2 * index:2 * index + 2]:
                np.testing.assert_array_equal(rendered, self.points[robot][:, [0, 2]])
            np.testing.assert_allclose(captured[4 + index], expected[robot][:, [0, 2]])
        np.testing.assert_array_equal(segments, [[[1, 2], [2, 4]]])
        trace = json.loads((self.root / "plots/test_trace.json").read_text())
        overlay = trace["ground_truth_overlay"]
        self.assertTrue(overlay["shared_by_all_robots"])
        self.assertFalse(overlay["per_robot_fitting"])
        self.assertEqual(trace["presentation"]["trajectory_alignment"], "none")
        self.assertEqual(trace["ate_evaluation"]["metrics"]["52"], metrics)
        self.assertGreater(metrics["joint_ate"], 1.0)
        lower, upper = (np.asarray(trace["presentation"]["panel_bounds"]["52"][key])
                        for key in ("lower", "upper"))
        for points in captured:
            self.assertTrue(np.all(points >= lower))
            self.assertTrue(np.all(points <= upper))

    def test_groundtruth_overlay_requires_groundtruth(self):
        with self.assertRaisesRegex(ValueError, "ground-truth directory"):
            RENDERER.render_fourteen_robot_stage_trajectories(
                REPO_ROOT, self.root / "plots", "test", self.root,
                show_groundtruth=True,
            )

    def prepare_centralized(self):
        self.provenance["command"].append("--run_explicit_anchor_centralized=true")
        self.save_provenance()
        (self.dpgo / "centralized_explicit_anchors.csv").write_bytes(
            (self.dpgo / "cbs_iteration_52.csv").read_bytes()
        )
        (self.dpgo / "cbs_trajectory_snapshots.csv").unlink()

    def render_centralized(self, **kwargs):
        RENDERER.render_fourteen_robot_stage_trajectories(
            REPO_ROOT, self.root / "plots", "test", self.root,
            expected_robot_count=2, reference_csv=self.reference,
            solution="centralized_explicit_anchors", **kwargs,
        )

    def test_centralized_uses_saved_solution_without_fabricating_cbs_iteration(self):
        self.prepare_centralized()
        self.render_centralized(show_loop_edges=True)
        trace = json.loads((self.root / "plots/test_trace.json").read_text())
        self.assertEqual(trace["iterations"], [])
        self.assertIsNone(trace["stage_boundaries"])
        self.assertNotIn("round_counts_at_snapshot", trace)
        self.assertEqual(trace["solution"], "centralized_explicit_anchors")
        self.assertEqual(trace["presentation"]["trajectory_alignment"], "none")
        self.assertTrue((self.root / "plots/test.pdf").is_file())
        self.assertEqual(trace["loop_overlay"]["edge_count"], 1)

    def test_centralized_rejects_cbs_iteration(self):
        with self.assertRaisesRegex(ValueError, "no CBS snapshot"):
            self.render_centralized(selected_iteration=52)

    def test_centralized_requires_enabled_solver_in_provenance(self):
        self.prepare_centralized()
        self.provenance["command"][-1] = "--run_explicit_anchor_centralized=false"
        self.save_provenance()
        with self.assertRaisesRegex(ValueError, "unexpected stage snapshot configuration"):
            self.render_centralized()

    def test_centralized_rejects_wrong_keyframe_identities(self):
        self.prepare_centralized()
        path = self.dpgo / "centralized_explicit_anchors.csv"
        path.write_text(path.read_text().replace("0,robot0,0,", "999,robot0,0,"))
        with self.assertRaisesRegex(ValueError, "keyframe identities differ"):
            self.render_centralized()


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dpvo.loop_closure.centralized import Sim3
from dpvo.loop_closure.offline_dpgo import (
    _parser,
    build_command,
    run as run_dpgo,
    validate_dpgo_input,
)
from dpvo.loop_closure.pose_graph import (
    PoseGraphEdge,
    PoseGraphVertex,
    Sim3PoseGraph,
    read_json,
    write_json,
)
from dpvo.loop_closure.tracking_artifact import (
    load_artifact_root,
    load_tracking_artifact,
    save_tracking_artifact,
    write_raw_pose_graph,
)


def _poses(count, step=1.0):
    poses = np.zeros((count, 7), dtype=np.float32)
    poses[:, 0] = np.arange(count) * step
    poses[:, 6] = 1.0
    return poses


def _save_artifact(root: Path, robot_id: str, step: float = 1.0):
    output = root / robot_id
    frames = output / "frames"
    frames.mkdir(parents=True)
    for input_index in (0, 2, 5):
        (frames / f"{input_index:08d}.jpg").write_bytes(
            f"{robot_id}-{input_index}".encode()
        )
    save_tracking_artifact(
        output,
        robot_id=robot_id,
        session_id=f"{robot_id}_tracking",
        keyframe_input_indices=[0, 2, 5],
        keyframe_timestamps=[0.0, 0.2, 0.5],
        keyframe_poses_xyzw=_poses(3, step),
        internal_poses_xyzw=_poses(3, step),
        internal_intrinsics=np.tile([100.0, 100.0, 80.0, 60.0], (3, 1)),
        patch_disparities=[1.0, 1.1, 1.2],
        session_from_map=np.eye(4),
        input_poses_xyzw=_poses(6, step),
        map_points=np.arange(18, dtype=np.float32).reshape(6, 3),
        map_colors=np.full((6, 3), 127, dtype=np.uint8),
        patches_per_keyframe=2,
        image_width=160,
        image_height=128,
        dpvo_resolution=4,
        input_frame_count=6,
        random_seed=1234,
        config_path="config/default.yaml",
        network_path="dpvo.pth",
    )


def _verified_graph(connected=True):
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
    pairs = [(0, 1), (1, 2)] if connected else [(0, 1)]
    edges = [
        PoseGraphEdge(
            edge_id=index,
            source=source,
            target=target,
            measurement=Sim3.identity(),
            information_diagonal=np.ones(7),
            edge_type="inter_robot_loop_closure",
        )
        for index, (source, target) in enumerate(pairs)
    ]
    return Sim3PoseGraph(
        name="verified raw graph",
        vertices=vertices,
        edges=edges,
        metadata={
            "pipeline_stage": "geometric_verification",
            "input_contains_global_optimization": False,
            "initialization": "per_robot_local_map_original_scale",
        },
    )


class TrackingArtifactTest(unittest.TestCase):
    def test_roundtrip_and_raw_per_robot_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tracking"
            _save_artifact(root, "robot0", 1.0)
            _save_artifact(root, "robot1", 2.0)
            artifacts = load_artifact_root(root)
            self.assertEqual(set(artifacts), {"robot0", "robot1"})
            self.assertEqual(artifacts["robot1"].keyframe_count, 3)
            self.assertEqual(
                artifacts["robot1"].frame_path(2).name,
                "00000005.jpg",
            )
            self.assertEqual(artifacts["robot1"].map_points.shape, (6, 3))

            output = root / "unoptimized_tracking_graph"
            graph = write_raw_pose_graph(artifacts, output)
            self.assertEqual(len(graph.vertices), 6)
            self.assertEqual(len(graph.edges), 4)
            self.assertFalse(
                any(
                    edge.edge_type == "inter_robot_loop_closure"
                    for edge in graph.edges
                )
            )
            self.assertEqual(
                read_json(output.with_name(f"{output.name}_robot1.json"))
                .metadata["coordinate_frame"],
                "robot_local_map",
            )

    def test_image_tampering_invalidates_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "robot0"
            _save_artifact(root.parent, root.name)
            (root / "frames" / "00000002.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "image checksum mismatch"):
                load_tracking_artifact(root)

    def test_map_tampering_invalidates_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "robot0"
            _save_artifact(root.parent, root.name)
            (root / "map.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "map checksum mismatch"):
                load_tracking_artifact(root)

    def test_robot_id_filtering_and_default_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tracking"
            for robot_id in ("robot0", "robot1", "robot2"):
                _save_artifact(root, robot_id)

            self.assertEqual(
                set(load_artifact_root(root)),
                {"robot0", "robot1", "robot2"},
            )
            self.assertEqual(
                set(load_artifact_root(root, ["robot0", "robot2"])),
                {"robot0", "robot2"},
            )
            with self.assertRaisesRegex(ValueError, "duplicate robot ids"):
                load_artifact_root(root, ["robot0", "robot0"])
            with self.assertRaisesRegex(ValueError, "unknown or missing"):
                load_artifact_root(root, ["robot0", "robot9"])



class OfflineDpgoBoundaryTest(unittest.TestCase):
    def test_default_schedule_is_alternating_twenty_round_blocks(self):
        args = _parser().parse_args(
            [
                "--input-graph",
                "input.json",
                "--output-dir",
                "output",
                "--cbs-executable",
                "cbs_dpvo_sim3_offline",
            ]
        )
        self.assertEqual(args.stage_mode, "alternating")
        self.assertEqual(args.pose_warmup_iterations, 0)
        self.assertEqual(args.pose_block_iterations, 20)
        self.assertEqual(args.anchor_block_iterations, 20)
        self.assertEqual(args.target_hellinger, 0.1)
        self.assertTrue(args.hellinger_quadratic_term)

    def test_no_covariance_transport_baseline_is_explicit(self):
        args = _parser().parse_args(
            [
                "--input-graph",
                "input.json",
                "--output-dir",
                "output",
                "--cbs-executable",
                "cbs_dpvo_sim3_offline",
                "--sim3-covariance-transport",
                "none",
            ]
        )
        self.assertEqual(args.sim3_covariance_transport, "none")

    def test_hellinger_quadratic_term_can_be_disabled_for_control(self):
        args = _parser().parse_args(
            [
                "--input-graph",
                "input.json",
                "--output-dir",
                "output",
                "--cbs-executable",
                "cbs_dpvo_sim3_offline",
                "--no-hellinger-quadratic-term",
            ]
        )
        self.assertFalse(args.hellinger_quadratic_term)

    def test_only_connected_unoptimized_verified_graph_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = root / "accepted.json"
            write_json(_verified_graph(), accepted)
            graph, loop_count = validate_dpgo_input(accepted)
            self.assertEqual(loop_count, 2)
            self.assertTrue(
                all(vertex.optimized_estimate is None for vertex in graph.vertices)
            )

            disconnected = root / "disconnected.json"
            write_json(_verified_graph(connected=False), disconnected)
            with self.assertRaisesRegex(ValueError, "does not connect every robot"):
                validate_dpgo_input(disconnected)

            contaminated_graph = _verified_graph()
            contaminated_graph.vertices[1].optimized_estimate = Sim3.identity()
            contaminated = root / "contaminated.json"
            write_json(contaminated_graph, contaminated)
            with self.assertRaisesRegex(ValueError, "optimized vertex"):
                validate_dpgo_input(contaminated)

    def test_legacy_unoptimized_input_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            graph = _verified_graph()
            graph.metadata = {
                "initialization": "per_robot_local_map_original_scale"
            }
            write_json(graph, path)
            with self.assertRaisesRegex(ValueError, "legacy raw exports require"):
                validate_dpgo_input(path)
            accepted, loop_count = validate_dpgo_input(
                path, allow_legacy_unoptimized_input=True
            )
            self.assertEqual(loop_count, 2)
            self.assertTrue(
                all(vertex.optimized_estimate is None for vertex in accepted.vertices)
            )

    def test_command_runs_cbs_and_both_centralized_paths_from_one_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "cbs_dpvo_sim3_offline"
            executable.write_text("test")
            args = argparse.Namespace(
                cbs_executable=str(executable),
                iterations=1000,
                stage_mode="alternating",
                anchor_start_iteration=30,
                anchor_stage_probability=0.5,
                pose_warmup_iterations=0,
                pose_block_iterations=20,
                anchor_block_iterations=20,
                target_hellinger=0.1,
                hellinger_quadratic_term=True,
                contract_alpha=0.95,
                d_reset=0.6,
                sim3_covariance_transport="bernoulli",
                bootstrap_robot_anchors=False,
                odom_scale_sigma=-1.0,
                inter_loop_scale_sigma=-1.0,
                random_seed=42,
                huber_k=-1.0,
                centralized_max_iterations=300,
                write_rerun_rrd=True,
                rerun_iteration_stride=1,
                rerun_stream=False,
                rerun_url="rerun+http://127.0.0.1:9876/proxy",
                allow_legacy_unoptimized_input=False,
            )
            command = build_command(args, root / "input.json", root / "output")
            self.assertIn("--run_cbs=true", command)
            self.assertIn("--run_centralized=true", command)
            self.assertIn("--run_explicit_anchor_centralized=true", command)
            self.assertIn("--target_hellinger=0.1", command)
            self.assertIn("--hellinger_quadratic_term=true", command)
            self.assertIn("--stage_mode=alternating", command)
            self.assertIn("--pose_warmup_iterations=0", command)
            self.assertIn("--pose_block_iterations=20", command)
            self.assertIn("--anchor_block_iterations=20", command)
            self.assertIn("--sim3_covariance_transport=bernoulli", command)
            self.assertIn("--bootstrap_robot_anchors=false", command)
            self.assertIn("--rerun_iteration_stride=1", command)

            source = root / "verified.json"
            write_json(_verified_graph(), source)
            args.input_graph = str(source)
            args.output_dir = str(root / "output")
            args.dry_run = True
            run_dpgo(args)
            provenance = json.loads(
                (root / "output" / "offline_dpgo_provenance.json").read_text()
            )
            self.assertEqual(
                provenance["source_sha256"],
                provenance["copied_input_sha256"],
            )
            self.assertTrue(provenance["copied_input_is_byte_identical"])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from dpvo.loop_closure.centralized import (
    CentralizedPgoResult,
    RobotMapConstraint,
    Sim3,
)
from dpvo.loop_closure.pose_graph import (
    PoseGraphEdge,
    PoseGraphVertex,
    Sim3PoseGraph,
    build_keyframe_graph,
    optimize_pose_graph,
    read_json,
    restore_original_robot_scales,
    sim3_log,
    split_keyframe_graph_by_robot,
    write_g2o,
    write_json,
)


def _pose(x, y=0.0, z=0.0):
    return Sim3([x, y, z], np.eye(3), 1.0)


def _sim3_exp(vector):
    omega = vector[:3]
    algebra = np.zeros((4, 4))
    algebra[:3, :3] = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    ) + vector[6] * np.eye(3)
    algebra[:3, 3] = vector[3:6]
    matrix = expm(algebra)
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    return Sim3(matrix[:3, 3], matrix[:3, :3] / scale, scale)


class PoseGraphIoTest(unittest.TestCase):
    def test_g2o_log_is_a_true_sim3_lie_logarithm(self):
        transform = Sim3(
            [1.2, -0.4, 0.7],
            Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix(),
            1.35,
        )
        vector = sim3_log(transform)
        expected = np.eye(4)
        expected[:3, :3] = transform.scale * transform.rotation
        expected[:3, 3] = transform.translation
        recovered = _sim3_exp(vector)
        recovered_matrix = np.eye(4)
        recovered_matrix[:3, :3] = recovered.scale * recovered.rotation
        recovered_matrix[:3, 3] = recovered.translation
        np.testing.assert_allclose(recovered_matrix, expected, atol=1e-10)

    def test_json_roundtrip_and_g2o_export(self):
        graph = Sim3PoseGraph(
            "test graph",
            vertices=[
                PoseGraphVertex(0, "robot0", "run", Sim3.identity(), fixed=True),
                PoseGraphVertex(1, "robot1", "run", _pose(2.0)),
            ],
            edges=[
                PoseGraphEdge(
                    0,
                    0,
                    1,
                    _pose(-2.0),
                    np.arange(1.0, 8.0),
                    "inter_robot_loop_closure",
                    {"inliers": 42},
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "graph.json"
            g2o_path = Path(directory) / "graph.g2o"
            write_json(graph, json_path)
            write_g2o(graph, g2o_path)
            loaded = read_json(json_path)

            self.assertEqual(loaded.name, graph.name)
            self.assertEqual(loaded.edges[0].metadata["inliers"], 42)
            np.testing.assert_allclose(
                loaded.edges[0].information_diagonal,
                np.arange(1.0, 8.0),
            )
            text = g2o_path.read_text()
            self.assertEqual(text.count("VERTEX_SIM3:EXPMAP"), 2)
            self.assertEqual(text.count("EDGE_SIM3:EXPMAP"), 1)
            self.assertIn("FIX 0", text)

            # Mirror g2o's file readers and EdgeSim3::computeError(). Vertex
            # lines are inverted on read, as are edge measurements.
            vertex_lines = [
                line for line in text.splitlines() if line.startswith("VERTEX_")
            ]
            edge_line = next(
                line for line in text.splitlines() if line.startswith("EDGE_SIM3")
            )
            g2o_vertices = {
                int(parts[1]): _sim3_exp(np.asarray(parts[2:9], dtype=float)).inverse()
                for parts in (line.split() for line in vertex_lines)
            }
            parts = edge_line.split()
            g2o_measurement = _sim3_exp(
                np.asarray(parts[3:10], dtype=float)
            ).inverse()
            error = (
                g2o_measurement
                .compose(g2o_vertices[0])
                .compose(g2o_vertices[1].inverse())
            )
            np.testing.assert_allclose(error.translation, np.zeros(3), atol=1e-10)
            np.testing.assert_allclose(error.rotation, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(error.scale, 1.0, places=10)

    def test_keyframe_graph_contains_odometry_and_loop_edges(self):
        robot0 = ("robot0", "run0")
        robot1 = ("robot1", "run1")
        query_pose = _pose(1.0)
        match_pose = _pose(2.0)
        loop = RobotMapConstraint(
            robot0,
            robot1,
            query_pose,
            match_pose,
            _pose(0.25),
            weight=2.0,
            query_keyframe_id=1,
            match_keyframe_id=2,
            inliers=60,
        )
        result = CentralizedPgoResult(
            {robot0: Sim3.identity(), robot1: _pose(5.0)},
            True,
            0.0,
            0.0,
        )
        graph = build_keyframe_graph(
            [loop],
            result,
            {
                "robot0": [_pose(0.0), query_pose],
                "robot1": [_pose(0.0), _pose(1.0), match_pose],
            },
            {"robot0": "run0", "robot1": "run1"},
            "robot0",
            odometry_weight=50.0,
            timestamps={
                "robot0": [1.0, 2.0],
                "robot1": [3.0, 4.0, 5.0],
            },
        )

        self.assertEqual(len(graph.vertices), 5)
        self.assertEqual(
            [edge.edge_type for edge in graph.edges].count("visual_odometry"),
            3,
        )
        loop_edge = next(
            edge for edge in graph.edges if edge.edge_type == "inter_robot_loop_closure"
        )
        np.testing.assert_allclose(loop_edge.measurement.translation, [0.25, 0, 0])
        self.assertEqual(loop_edge.metadata["inliers"], 60)
        self.assertEqual([vertex.timestamp for vertex in graph.vertices], [1, 2, 3, 4, 5])

    def test_unoptimized_and_per_robot_graphs_keep_original_scale(self):
        robot0 = ("robot0", "run0")
        robot1 = ("robot1", "run1")
        map_transform = Sim3(
            [4.0, -1.0, 0.5],
            Rotation.from_euler("z", 0.3).as_matrix(),
            0.2,
        )
        local_paths = {
            "robot0": [_pose(0.0), _pose(1.0)],
            "robot1": [_pose(0.0), _pose(3.0), _pose(5.0)],
        }
        result = CentralizedPgoResult(
            {robot0: Sim3.identity(), robot1: map_transform},
            True,
            0.0,
            0.0,
        )
        aligned = build_keyframe_graph(
            [],
            result,
            local_paths,
            {"robot0": "run0", "robot1": "run1"},
            "robot0",
        )
        raw = restore_original_robot_scales(aligned, result.transforms)

        self.assertEqual(
            raw.metadata["initialization"],
            "per_robot_local_map_original_scale",
        )
        robot1_vertices = [
            vertex for vertex in raw.vertices if vertex.robot_id == "robot1"
        ]
        self.assertTrue(all(vertex.estimate.scale == 1.0 for vertex in raw.vertices))
        np.testing.assert_allclose(
            [vertex.estimate.translation[0] for vertex in robot1_vertices],
            [0.0, 3.0, 5.0],
            atol=1e-12,
        )

        per_robot = split_keyframe_graph_by_robot(raw)
        self.assertEqual(set(per_robot), {"robot0", "robot1"})
        self.assertEqual(len(per_robot["robot0"].vertices), 2)
        self.assertEqual(len(per_robot["robot0"].edges), 1)
        self.assertEqual(len(per_robot["robot1"].vertices), 3)
        self.assertEqual(len(per_robot["robot1"].edges), 2)
        self.assertTrue(per_robot["robot1"].vertices[0].fixed)
        self.assertFalse(per_robot["robot1"].vertices[1].fixed)

    def test_offline_optimizer_recovers_consistent_vertices(self):
        truth = {
            0: Sim3.identity(),
            1: Sim3(
                [2.0, -0.5, 0.2],
                Rotation.from_euler("z", 0.2).as_matrix(),
                1.2,
            ),
            2: Sim3(
                [-1.0, 1.5, -0.1],
                Rotation.from_euler("xyz", [0.1, 0.0, -0.1]).as_matrix(),
                0.9,
            ),
        }
        graph = Sim3PoseGraph(
            "optimization test",
            vertices=[
                PoseGraphVertex(
                    vertex_id,
                    f"robot{vertex_id}",
                    "run",
                    Sim3.identity(),
                    fixed=vertex_id == 0,
                )
                for vertex_id in truth
            ],
            edges=[
                PoseGraphEdge(
                    index,
                    source,
                    target,
                    truth[target].inverse().compose(truth[source]),
                    np.ones(7),
                    "test",
                )
                for index, (source, target) in enumerate(((0, 1), (1, 2), (0, 2)))
            ],
        )

        result = optimize_pose_graph(graph, loss="linear")

        self.assertTrue(result.success)
        self.assertLess(result.residual_norm, 1e-6)
        for vertex in graph.vertices:
            np.testing.assert_allclose(
                vertex.optimized_estimate.translation,
                truth[vertex.vertex_id].translation,
                atol=1e-5,
            )
            self.assertAlmostEqual(
                vertex.optimized_estimate.scale,
                truth[vertex.vertex_id].scale,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()

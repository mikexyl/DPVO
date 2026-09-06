import tempfile
import unittest
from pathlib import Path

import numpy as np

from dpvo.loop_closure.centralized import Sim3
from dpvo.loop_closure.pose_graph import (
    PoseGraphEdge,
    PoseGraphVertex,
    Sim3PoseGraph,
    read_json,
    write_json,
)
from dpvo.loop_closure.pose_graph_noise import (
    NOISE_METADATA_KEY,
    perturb_loop_measurements,
)


class PoseGraphNoiseTest(unittest.TestCase):
    def make_graph(self):
        vertices = [
            PoseGraphVertex(0, "robot0", "session0", Sim3.identity(), fixed=True),
            PoseGraphVertex(1, "robot0", "session0", Sim3.identity()),
            PoseGraphVertex(2, "robot1", "session1", Sim3.identity(), fixed=True),
        ]
        edges = [
            PoseGraphEdge(
                0,
                0,
                1,
                Sim3.identity(),
                np.full(7, 100.0),
                "visual_odometry",
            ),
            PoseGraphEdge(
                1,
                1,
                2,
                Sim3([0.2, -0.1, 0.3], np.eye(3), 1.2),
                np.full(7, 0.5),
                "inter_robot_loop_closure",
                {"query_to_match": {}},
            ),
        ]
        return Sim3PoseGraph("test graph", vertices, edges)

    def test_only_loop_measurements_change_deterministically(self):
        first, first_records = perturb_loop_measurements(
            self.make_graph(),
            translation_sigma=0.05,
            rotation_sigma_degrees=3.0,
            log_scale_sigma=0.05,
            seed=42,
        )
        second, second_records = perturb_loop_measurements(
            self.make_graph(),
            translation_sigma=0.05,
            rotation_sigma_degrees=3.0,
            log_scale_sigma=0.05,
            seed=42,
        )

        np.testing.assert_allclose(first.edges[0].measurement.to_vector(), 0.0)
        np.testing.assert_allclose(
            first.edges[0].information_diagonal,
            self.make_graph().edges[0].information_diagonal,
        )
        np.testing.assert_allclose(
            first.edges[1].measurement.to_vector(),
            second.edges[1].measurement.to_vector(),
        )
        np.testing.assert_allclose(
            first.edges[1].information_diagonal,
            self.make_graph().edges[1].information_diagonal,
        )
        self.assertNotEqual(
            first.edges[1].measurement.to_vector().tolist(),
            self.make_graph().edges[1].measurement.to_vector().tolist(),
        )
        self.assertEqual(first_records, second_records)
        self.assertEqual(
            first.metadata[NOISE_METADATA_KEY]["perturbed_edge_count"],
            1,
        )
        self.assertEqual(
            first.edges[1].metadata["query_to_match"],
            first_records[0]["perturbed_measurement"],
        )

    def test_json_round_trip_retains_noise_provenance(self):
        graph, _ = perturb_loop_measurements(
            self.make_graph(),
            translation_sigma=0.05,
            rotation_sigma_degrees=3.0,
            log_scale_sigma=0.05,
            seed=42,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noisy.json"
            write_json(graph, path)
            loaded = read_json(path)
        self.assertEqual(
            loaded.metadata[NOISE_METADATA_KEY],
            graph.metadata[NOISE_METADATA_KEY],
        )
        self.assertEqual(
            loaded.edges[1].metadata[NOISE_METADATA_KEY],
            graph.edges[1].metadata[NOISE_METADATA_KEY],
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dpvo.loop_closure.centralized import (
    CentralizedRobotMapPGO,
    RobotMapConstraint,
    Sim3,
)


class CentralizedRobotMapPgoTest(unittest.TestCase):
    def test_recovers_three_robot_map_alignment(self):
        robots = [("robot0", "run"), ("robot1", "run"), ("robot2", "run")]
        alignments = {
            robots[0]: Sim3.identity(),
            robots[1]: Sim3(
                [2.0, -1.0, 0.4],
                Rotation.from_euler("z", 0.25).as_matrix(),
                1.15,
            ),
            robots[2]: Sim3(
                [-1.0, 1.5, -0.2],
                Rotation.from_euler("xyz", [0.1, -0.05, -0.2]).as_matrix(),
                0.9,
            ),
        }
        poses = {
            robots[0]: Sim3.from_pose([0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0]),
            robots[1]: Sim3.from_pose([1.2, -0.4, 0.2, 0.0, 0.0, 0.1, 0.995]),
            robots[2]: Sim3.from_pose([-0.8, 0.7, 0.1, 0.0, 0.05, 0.0, 0.999]),
        }

        constraints = []
        for query, match in ((robots[0], robots[1]), (robots[1], robots[2])):
            query_to_match = (
                poses[match].inverse()
                .compose(alignments[match].inverse())
                .compose(alignments[query])
                .compose(poses[query])
            )
            constraints.append(
                RobotMapConstraint(
                    query,
                    match,
                    poses[query],
                    poses[match],
                    query_to_match,
                )
            )

        result = CentralizedRobotMapPGO(anchor=robots[0]).solve(constraints)

        self.assertTrue(result.success)
        self.assertLess(result.residual_norm, 1e-7)
        for robot in robots:
            np.testing.assert_allclose(
                result.transforms[robot].translation,
                alignments[robot].translation,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                result.transforms[robot].rotation,
                alignments[robot].rotation,
                atol=1e-6,
            )
            self.assertAlmostEqual(
                result.transforms[robot].scale,
                alignments[robot].scale,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()

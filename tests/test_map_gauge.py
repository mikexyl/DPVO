import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dpvo.map_gauge import (
    normalization_change,
    scale_camera_points,
    similarity_scale,
    transform_points,
    transform_pose_xyzw,
)


class StableMapGaugeTest(unittest.TestCase):
    def test_normalization_change_recovers_old_map_points(self):
        rotation = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
        translation = np.array([0.4, -0.8, 0.2])
        camera_to_old_map = np.eye(4)
        camera_to_old_map[:3, :3] = rotation.T
        camera_to_old_map[:3, 3] = -rotation.T @ translation
        disparity_scale = 2.5
        old_points = np.array([[0.2, 0.4, 1.0], [-1.0, 0.5, 2.0]])
        new_points = disparity_scale * (
            old_points @ rotation.T + translation
        )

        old_from_new = normalization_change(
            camera_to_old_map,
            disparity_scale,
        )

        np.testing.assert_allclose(
            transform_points(old_from_new, new_points),
            old_points,
            atol=1e-12,
        )
        self.assertAlmostEqual(similarity_scale(old_from_new), 0.4)

    def test_pose_and_camera_points_use_the_same_stable_length_unit(self):
        gauge = np.eye(4)
        gauge_rotation = Rotation.from_euler("z", 0.35).as_matrix()
        gauge[:3, :3] = 1.7 * gauge_rotation
        gauge[:3, 3] = [0.2, -0.4, 0.8]
        pose = np.r_[
            [0.5, 0.1, -0.2],
            Rotation.from_euler("y", -0.2).as_quat(),
        ]
        camera_points = np.array([[0.1, 0.2, 2.0], [-0.2, 0.3, 1.4]])

        stable_pose = transform_pose_xyzw(gauge, pose)
        stable_camera_points = scale_camera_points(gauge, camera_points)
        current_map_points = (
            camera_points @ Rotation.from_quat(pose[3:]).as_matrix().T
            + pose[:3]
        )
        expected = transform_points(gauge, current_map_points)
        actual = (
            stable_camera_points
            @ Rotation.from_quat(stable_pose[3:]).as_matrix().T
            + stable_pose[:3]
        )

        np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

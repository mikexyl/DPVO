"""Exercise the actual Viser API without a camera or GPU."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dpvo.viser_viewer import ViserViewer


class ViserViewerTests(unittest.TestCase):
    def test_scene_coordinates_color_limits_and_reset(self):
        graph = SimpleNamespace(
            poses_=torch.tensor([[0., 0, 0, 0, 0, 0, 1], [1., 0, 0, 0, 0, 0, 1]]),
            points_=torch.arange(300, dtype=torch.float32).reshape(100, 3),
            colors_=torch.full((100, 3), 200, dtype=torch.uint8))
        viewer = ViserViewer(graph, 24, 32, web_port=19091, max_points=10, dense_controls=True)
        try:
            image = torch.zeros(3, 24, 32, dtype=torch.uint8)
            image[2] = 255  # BGR red must be RGB red in the browser.
            viewer.update_image(image)
            intrinsics = torch.tensor([20., 20, 16, 12])
            viewer.update_state(intrinsics, 2, 100)
            np.testing.assert_array_equal(viewer.image[0, 0], [255, 0, 0])
            np.testing.assert_allclose(viewer.camera.position, [-1, 0, 0])
            np.testing.assert_allclose(viewer.camera.wxyz, [1, 0, 0, 0])
            self.assertLessEqual(len(viewer.points.points), 10)
            self.assertTrue(viewer.camera.visible)
            original = viewer.points
            viewer.update_state(intrinsics, 2, 100)
            self.assertIs(viewer.points, original)
            self.assertFalse(viewer.dense_enabled.value)
            viewer.update_dense(np.array([[1., 2., 3.]]), np.array([[10, 20, 30]], np.uint8))
            self.assertTrue(viewer.dense_points.visible)
            viewer.reset()
            self.assertFalse(viewer.dense_points.visible)
            self.assertFalse(viewer.camera.visible)
            self.assertFalse(viewer.points.visible)
            self.assertFalse(viewer.trajectory.visible)
            np.testing.assert_array_equal(viewer.center, np.zeros(3))
            viewer.request_tracking(False)
            self.assertIs(viewer.take_command(), False)
            self.assertIsNone(viewer.take_command())
            viewer.set_running(False)
            self.assertFalse(viewer.start_button.disabled)
            self.assertTrue(viewer.stop_button.disabled)
            viewer.request_tracking(True)
            self.assertIs(viewer.take_command(), True)
            viewer.set_running(True)
            self.assertTrue(viewer.start_button.disabled)
            self.assertFalse(viewer.stop_button.disabled)
            self.assertTrue(viewer.dense_enabled.disabled)
        finally:
            viewer.join()


if __name__ == '__main__':
    unittest.main()

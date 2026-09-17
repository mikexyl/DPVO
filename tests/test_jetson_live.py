"""Run in dpvo:jetson-live (RealSense SDK and current Rerun are required)."""
import tempfile
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pyrealsense2 as rs
import torch

from deploy.jetson.realsense_camera import LatestFrame, camera_maps, color_to_bgr
HAS_RERUN = importlib.util.find_spec('rerun') is not None
if HAS_RERUN:
    from dpvo.rerun_viewer import RerunViewer
from deploy.jetson.tracking_health import tracking_health


class CaptureStrideTests(unittest.TestCase):
    def test_native_yuyv_black_white_pair_becomes_bgr(self):
        raw = np.array([[[16, 128], [235, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(color_to_bgr(raw, 'yuyv'),
                                      [[[0, 0, 0], [255, 255, 255]]])
        for packed in (raw.reshape(1, 4), raw.view(np.uint16).reshape(1, 2)):
            np.testing.assert_array_equal(color_to_bgr(packed, 'yuyv'),
                                          [[[0, 0, 0], [255, 255, 255]]])
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertIs(color_to_bgr(bgr, 'bgr8'), bgr)
        with self.assertRaises(ValueError):
            color_to_bgr(raw, 'invalid')

    def test_stride_precedes_image_copy_and_preserves_capture_timestamps(self):
        copied = []
        exhausted = threading.Event()
        release = threading.Event()

        class Pipeline:
            index = 0

            def wait_for_frames(self, timeout):
                self.index += 1
                if self.index > 9:
                    exhausted.set()
                    release.wait(2)
                    raise RuntimeError('Test camera stopped')
                number = self.index

                def data():
                    copied.append(number)
                    return np.zeros((2, 2, 3), dtype=np.uint8)

                color = SimpleNamespace(get_data=data, get_timestamp=lambda: number * 1000 / 30)
                return SimpleNamespace(get_color_frame=lambda: color)

        capture = LatestFrame(Pipeline(), stride=3)
        try:
            self.assertTrue(exhausted.wait(2))
            number, timestamp, _, image = capture.get(0)
            self.assertEqual(copied, [1, 4, 7])
            self.assertEqual(number, 7)  # Only the latest selected frame is retained.
            self.assertAlmostEqual(timestamp, 7 / 30)
            self.assertEqual(image.shape, (2, 2, 3))
        finally:
            capture.stop.set()
            release.set()
            capture.thread.join(2)

    def test_invalid_stride_rejected(self):
        with self.assertRaises(ValueError):
            LatestFrame(None, stride=0)


class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width, self.intrinsics.height = 1280, 800
        self.intrinsics.fx = self.intrinsics.fy = 644
        self.intrinsics.ppx, self.intrinsics.ppy = 642, 399
        self.intrinsics.model = rs.distortion.inverse_brown_conrady
        self.intrinsics.coeffs = [-0.0547, 0.0624, -0.00035, -0.00012, -0.0201]

    def test_rectified_pixels_deproject_to_expected_rays(self):
        maps, (fx, fy, cx, cy) = camera_maps(self.intrinsics, 384, 240)
        for x, y in [(0, 0), (383, 239), (10, 230), (370, 12), (190, 120)]:
            source = [float(maps[0][y, x]), float(maps[1][y, x])]
            ray = rs.rs2_deproject_pixel_to_point(self.intrinsics, source, 1)
            np.testing.assert_allclose(ray, [(x-cx)/fx, (y-cy)/fy, 1], atol=1e-5)

    def test_resize_preserves_full_nominal_field_of_view(self):
        _, values = camera_maps(self.intrinsics, 384, 240)
        np.testing.assert_allclose(values, np.array([644, 644, 642, 399]) * 0.3)
        with self.assertRaises(ValueError):
            camera_maps(self.intrinsics, 368, 240)


class TrackingHealthTests(unittest.TestCase):
    def test_finite_pose_with_collapsed_depth_is_tracking_loss(self):
        # Regression: the observed diverged live pose passed the old finite-only check.
        poses = torch.tensor([[75424., 10039., 31851., -0.415, -0.627, 0.157, 0.640]])
        health = tracking_health(poses, torch.full((3, 32, 3, 3), 1e-4), True)
        self.assertTrue(health['finite'])
        self.assertTrue(health['lost'])

    def test_healthy_depth_and_uninitialized_graph_are_not_collapse(self):
        poses = torch.tensor([[0., 0, 0, 0, 0, 0, 1]])
        self.assertFalse(tracking_health(poses, torch.full((64,), 0.48), True)['lost'])
        self.assertFalse(tracking_health(poses, torch.zeros(64), False)['lost'])
        self.assertTrue(tracking_health(poses, torch.full((64,), float('nan')), True)['lost'])


@unittest.skipUnless(HAS_RERUN, 'Rerun is optional in the JetPack 6.2 Viser image')
class ViewerTests(unittest.TestCase):
    def test_conflicting_sinks_fail_before_initializing(self):
        with mock.patch('dpvo.rerun_viewer.rr.init') as init:
            with self.assertRaises(ValueError):
                RerunViewer(None, 240, 384, save_path='unused', web_port=9090)
            init.assert_not_called()

    def test_web_mode_does_not_spawn_native_viewer(self):
        with mock.patch('dpvo.rerun_viewer.rr') as rr:
            RerunViewer(None, 240, 384, web_port=19090, grpc_port=19876,
                        cors_allow_origin=['http://jetson:19090'])
            rr.serve_grpc.assert_called_once_with(
                grpc_port=19876, server_memory_limit='256MB',
                cors_allow_origin=['http://jetson:19090'])
            rr.serve_web_viewer.assert_called_once_with(web_port=19090, open_browser=False)
            rr.spawn.assert_not_called()

    def test_existing_scene_logs_with_current_rerun(self):
        graph = SimpleNamespace(
            poses_=torch.tensor([[0., 0, 0, 0, 0, 0, 1], [1., 0, 0, 0, 0, 0, 1]]),
            points_=torch.tensor([[0., 0, 2], [1., 1, 3]]),
            colors_=torch.tensor([[255, 0, 0], [0, 255, 0]], dtype=torch.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.rrd'
            viewer = RerunViewer(graph, 240, 384, save_path=path)
            viewer.update_image(torch.zeros(3, 240, 384, dtype=torch.uint8))
            viewer.update_state(torch.tensor([190., 190, 192, 120]), 2, 2)
            viewer.join()
            self.assertGreater(path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()

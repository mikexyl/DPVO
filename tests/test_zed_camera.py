"""Synthetic checks for the UVC left-eye geometry and mixed-camera fleet."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from deploy.jetson.zed_camera import camera_maps, left_image
from deploy.jetson.multi_robot.fleet import generate, load_fleet, PROFILES


class ZedCameraTest(unittest.TestCase):
    def test_left_eye_excludes_right_image(self):
        stereo = np.zeros((376, 1344, 3), dtype=np.uint8)
        stereo[:, 672:] = 255
        left = left_image(stereo)
        self.assertEqual(left.shape, (376, 672, 3))
        self.assertFalse(left.any())
        with self.assertRaises(ValueError):
            left_image(stereo[:, :672])

    def test_resize_intrinsics_and_full_image_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / 'camera.conf'
            calibration.write_text('[LEFT_CAM_VGA]\nfx=264\nfy=265\ncx=336\ncy=188\n'
                                   'k1=0\nk2=0\np1=0\np2=0\nk3=0\n')
            maps, intrinsics = camera_maps(calibration)
            np.testing.assert_allclose(intrinsics, [264*384/672, 265*240/376, 192, 120])
            self.assertAlmostEqual(float(maps[0][120, 192]), 336)
            self.assertAlmostEqual(float(maps[1][120, 192]), 188)
            self.assertAlmostEqual(float(maps[0][0, 383]), 383*672/384)
            self.assertAlmostEqual(float(maps[1][239, 0]), 239*376/240, places=4)

    def test_mixed_fleet_camera_parameters(self):
        config = load_fleet(PROFILES / 'fleet.yaml')
        with tempfile.TemporaryDirectory() as directory:
            generate(config, directory)
            params = yaml.safe_load((Path(directory) / 'robot2.worker.yaml').read_text())
            camera = params['/robot2/dpvo_camera']['ros__parameters']
            self.assertEqual(camera['camera_type'], 'zed')
            self.assertEqual(camera['camera_fps'] / camera['camera_stride'], 7.5)
            self.assertEqual(camera['camera_calibration'], '/models/SN29882942.conf')
            params = yaml.safe_load((Path(directory) / 'robot1.worker.yaml').read_text())
            self.assertEqual(params['/robot1/dpvo_camera']['ros__parameters']['camera_type'], 'realsense')


class ZedSelectionTest(unittest.TestCase):
    def test_serial_selects_capture_node_without_other_sensors(self):
        from deploy.jetson.multi_robot.select_zed import select
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usb = root / 'bus/usb/devices'
            video = root / 'class/video4linux'
            usb.mkdir(parents=True)
            video.mkdir(parents=True)
            for index, serial in enumerate(('29882942', '99999999')):
                hub = root / 'devices' / f'hub{index}'
                cam, hid = hub / 'camera', hub / 'hid'
                cam.mkdir(parents=True)
                hid.mkdir()
                (cam / 'idVendor').write_text('2b03\n')
                (hid / 'serial').write_text(serial)
                (usb / f'hid{index}').symlink_to(hid)
                interface = cam / 'interface'
                interface.mkdir()
                for channel in (0, 1):
                    node = video / f'video{index*2+channel}'
                    node.mkdir()
                    (node / 'index').write_text(str(channel))
                    (node / 'device').symlink_to(interface)
            self.assertEqual(select('29882942', root), '/dev/video0')
            self.assertEqual(select('99999999', root), '/dev/video2')
            with self.assertRaises(RuntimeError):
                select('', root)
            with self.assertRaises(RuntimeError):
                select('1234', root)

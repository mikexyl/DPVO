import unittest
from types import SimpleNamespace

import numpy as np

from ros2.dpvo_multi_robot.dpvo_multi_robot.cu_multi_core import (
    build_pinhole_rectification,
    calibration_from_camera_info,
    decode_raw_image,
    rectified_camera_info,
)


class CuMultiPlayerCoreTest(unittest.TestCase):
    @staticmethod
    def _info(distortion=None):
        return SimpleNamespace(
            width=4,
            height=2,
            k=[100.0, 0.0, 1.5, 0.0, 101.0, 0.5, 0.0, 0.0, 1.0],
            d=[0.0] * 5 if distortion is None else distortion,
            distortion_model="plumb_bob",
        )

    def test_rgb8_decode_handles_row_padding_and_channel_order(self):
        rgb = np.asarray(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        padded = np.zeros((2, 8), dtype=np.uint8)
        padded[:, :6] = rgb.reshape(2, 6)
        message = SimpleNamespace(
            width=2,
            height=2,
            encoding="rgb8",
            step=8,
            data=padded.tobytes(),
        )
        decoded, encoding = decode_raw_image(message)
        self.assertEqual(encoding, "bgr8")
        self.assertTrue(decoded.flags.c_contiguous)
        np.testing.assert_array_equal(decoded[0, 0], [3, 2, 1])
        np.testing.assert_array_equal(decoded[1, 1], [12, 11, 10])

    def test_zero_distortion_preserves_intrinsics_and_pixels(self):
        calibration = calibration_from_camera_info(self._info())
        rectification = build_pinhole_rectification(calibration)
        self.assertIsNone(rectification.map_x)
        np.testing.assert_array_equal(
            rectification.camera_matrix, calibration.camera_matrix
        )
        image = np.arange(8, dtype=np.uint8).reshape(2, 4)
        np.testing.assert_array_equal(rectification.rectify(image), image)
        fields = rectified_camera_info(rectification)
        self.assertEqual(fields["k"], self._info().k)
        self.assertEqual(fields["d"], [])

    def test_mono8_decode_preserves_graco_pixels_and_padding(self):
        padded = np.asarray([[1, 2, 99], [3, 4, 99]], dtype=np.uint8)
        message = SimpleNamespace(
            width=2,
            height=2,
            encoding="mono8",
            step=3,
            data=padded.tobytes(),
        )
        decoded, encoding = decode_raw_image(message)
        self.assertEqual(encoding, "mono8")
        self.assertTrue(decoded.flags.c_contiguous)
        np.testing.assert_array_equal(decoded, [[1, 2], [3, 4]])

    def test_nonzero_radtan_builds_rectification_maps(self):
        calibration = calibration_from_camera_info(
            self._info([-0.1, 0.01, 0.0, 0.0, 0.0])
        )
        rectification = build_pinhole_rectification(calibration)
        self.assertEqual(rectification.map_x.shape, (2, 4))
        self.assertEqual(rectification.map_y.shape, (2, 4))
        output = rectification.rectify(np.zeros((2, 4, 3), dtype=np.uint8))
        self.assertEqual(output.shape, (2, 4, 3))

    def test_invalid_encoding_and_truncated_data_fail(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            decode_raw_image(
                SimpleNamespace(
                    width=1, height=1, encoding="16UC1", step=2, data=b"\0\0"
                )
            )
        with self.assertRaisesRegex(ValueError, "expected at least"):
            decode_raw_image(
                SimpleNamespace(
                    width=2, height=2, encoding="mono8", step=2, data=b"\0"
                )
            )


if __name__ == "__main__":
    unittest.main()

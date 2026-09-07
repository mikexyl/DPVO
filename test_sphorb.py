import concurrent.futures
import gc
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from dpvo.sphorb import SphorbExtractor
from dpvo.sphere_bow import DESCRIPTOR_VERSIONS, panorama_uv, validate_descriptor_tag


def panorama_bearings(uv, width, height):
    longitude = ((uv[..., 0] + .5) / width - .5) * (2 * np.pi)
    latitude = ((uv[..., 1] + .5) / height - .5) * np.pi
    return np.stack((np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                     np.cos(latitude) * np.cos(longitude)), axis=-1)


class SphorbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.one = SphorbExtractor(workers=1)
        cls.four = SphorbExtractor(workers=4)
        rng = np.random.default_rng(12)
        cls.bgra = rng.integers(0, 256, (320, 640, 4), np.uint8)
        cls.bgra[..., 3] = 255
        cls.depth = np.full((320, 640), 3., np.float32)

    def assert_features_equal(self, a, b):
        for key in ('descriptors', 'uv', 'bearings', 'points', 'sizes', 'responses', 'octaves', 'orientations'):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)

    def test_coordinate_roundtrips_and_sizes(self):
        for width, height in ((1280, 640), (1024, 512), (333, 177), (2048, 1024)):
            bgra = cv2.resize(self.bgra, (width, height))
            f = self.four(bgra, np.ones((height, width), np.float32))
            self.assertGreater(len(f.descriptors), 100)
            self.assertTrue((f.uv[:, 0] >= -.5).all() and (f.uv[:, 0] < width - .5).all())
            self.assertTrue((f.uv[:, 1] >= -.5).all() and (f.uv[:, 1] <= height - .5).all())
            np.testing.assert_allclose(panorama_bearings(f.uv.astype(float), width, height), f.bearings, atol=5e-7)
            np.testing.assert_allclose(np.linalg.norm(f.bearings, axis=1), 1, atol=2e-7)
            expected = 31 * width / (5 * np.asarray([256, 204, 162, 128, 102, 80, 64])[f.octaves])
            np.testing.assert_allclose(f.sizes, expected, rtol=1e-7)
            uv = np.array([[-.5, -.5], [width - .50001, height - .5], [0, height / 2 - .5]])
            rays = panorama_bearings(uv, width, height)
            np.testing.assert_allclose(panorama_bearings(panorama_uv(rays, width, height), width, height), rays, atol=1e-7)

    def test_hidden_rgb_empty_masks_and_black(self):
        rng = np.random.default_rng(33)
        bgra = self.bgra.copy()
        # Both sides of the seam, a pole, and an interior hole.
        bgra[:35, :, 3] = 0
        bgra[:, :35, 3] = bgra[:, -35:, 3] = 0
        bgra[100:150, 280:330, 3] = 0
        a = self.four(bgra, self.depth)
        self.assertGreater(len(a.descriptors), 100)
        bgra[bgra[..., 3] == 0, :3] = rng.integers(0, 256, (np.count_nonzero(bgra[..., 3] == 0), 3), np.uint8)
        self.assert_features_equal(a, self.four(bgra, self.depth))
        bgra[..., 3] = 0
        self.assertEqual(self.four(bgra, self.depth).descriptors.shape, (0, 32))
        bgra[..., :3] = 0
        bgra[..., 3] = 255
        self.assertEqual(len(self.four(bgra, self.depth).descriptors), 0)
        # A black texture region remains valid: acceptance must not use intensity as validity.
        bgra[80:240, 100:500] = self.bgra[80:240, 100:500]
        uncapped = SphorbExtractor(features=100000, levels=1)
        full = uncapped(bgra, self.depth)
        bgra[np.all(bgra[..., :3] == 0, axis=-1), 3] = 0
        masked = uncapped(bgra, self.depth)
        self.assertGreater(len(full.descriptors), len(masked.descriptors))

    def test_depth_does_not_change_descriptors(self):
        a = self.four(self.bgra, self.depth)
        for invalid in (np.nan, np.inf, 0., -1.):
            b = self.four(self.bgra, np.full_like(self.depth, invalid))
            np.testing.assert_array_equal(a.descriptors, b.descriptors)
            self.assertTrue(np.isnan(b.points).all())
        np.testing.assert_allclose(np.linalg.norm(a.points, axis=1), 3, atol=1e-6)

    def test_repeat_instances_threading_and_ownership(self):
        a = self.one(self.bgra, self.depth)
        self.assert_features_equal(a, self.four(self.bgra, self.depth))
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(self.four, self.bgra, self.depth) for _ in range(3)]
            for future in futures:
                self.assert_features_equal(a, future.result())
        instance = SphorbExtractor()
        saved = instance(self.bgra, self.depth)
        del instance
        gc.collect()
        self.assert_features_equal(a, saved)
        saved.descriptors[:] = 0
        self.assert_features_equal(a, self.one(self.bgra, self.depth))
        with tempfile.TemporaryDirectory() as temp:
            saved.save(Path(temp) / 'features.npz')
            with np.load(Path(temp) / 'features.npz', allow_pickle=False) as data:
                self.assertNotIn('face_ids', data)
                self.assertEqual(data['descriptor_family'].item(), 'sphorb')

    def test_missing_corrupt_tables_and_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises((RuntimeError, ValueError)):
                SphorbExtractor(tables=temp)
            for path in self.one.tables.iterdir():
                shutil.copy2(path, Path(temp) / path.name)
            SphorbExtractor(tables=temp)
            path = Path(temp) / 'geoinfo256.pfm'
            data = bytearray(path.read_bytes())
            data[-50] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                SphorbExtractor(tables=temp)
        for kwargs in ({'levels': 0}, {'levels': 8}, {'threshold': 255}, {'features': 0}, {'workers': 0}):
            with self.assertRaises(ValueError):
                SphorbExtractor(**kwargs)

    def test_vocabulary_family_and_version(self):
        validate_descriptor_tag({}, 'cube-orb', DESCRIPTOR_VERSIONS['cube-orb'])
        for metadata in ({}, {'descriptor_family': 'cube-orb'},
                         {'descriptor_family': 'sphorb', 'descriptor_version': 'old'}):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                validate_descriptor_tag(metadata, 'sphorb', DESCRIPTOR_VERSIONS['sphorb'])


if __name__ == '__main__':
    unittest.main()

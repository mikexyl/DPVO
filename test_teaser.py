import importlib.util
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from dpvo.teaser import estimate_sim3, verify_teaser_geometry


@unittest.skipUnless(importlib.util.find_spec('dpvo._teaser'), 'run pixi run build-teaser')
class TeaserTests(unittest.TestCase):
    def fixture(self, n=80):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(n, 3))
        # Explicit seam/pole samples and the complete rear hemisphere.
        a[:6] = [[.001, 0, -2], [-.001, 0, -2], [0, 3, 0], [0, -3, 0], [2, 0, 0], [-2, 0, 0]]
        R = np.diag([-1., 1., -1.])
        t = np.array([.5, -.2, .3])
        return a, 1.8 * a @ R.T + t, R, t

    def test_full_sphere_sim3_and_output_ownership(self):
        a, b, R, t = self.fixture()
        one = estimate_sim3(a, b, noise_bound=.01, workers=1)
        saved = one['rotation'].copy()
        for workers in (4, 1, 4):
            got = estimate_sim3(a, b, noise_bound=.01, workers=workers)
            self.assertAlmostEqual(got['scale'], 1.8, places=7)
            np.testing.assert_allclose(got['rotation'], R, atol=1e-7)
            np.testing.assert_allclose(got['translation'], t, atol=1e-7)
        a[:] = 0
        np.testing.assert_array_equal(one['rotation'], saved)

    def test_outlier_recovery(self):
        a, b, R, t = self.fixture(100)
        b[20:] = np.random.default_rng(123).normal(size=(80, 3)) * 3
        got = estimate_sim3(a, b, noise_bound=.03)
        self.assertLess(abs(got['scale'] - 1.8), .02)
        np.testing.assert_allclose(got['rotation'], R, atol=.01)
        np.testing.assert_allclose(got['translation'], t, atol=.02)
        residual = np.linalg.norm(got['scale'] * a @ got['rotation'].T + got['translation'] - b, axis=1)
        self.assertTrue((residual[:20] < .03).all())
        self.assertFalse((residual[20:] < .03).any())

    def test_independent_sphere_frames_and_uv_unused(self):
        a, b, R, t = self.fixture()
        q, c = SimpleNamespace(points=a, uv=np.zeros((len(a),2))), SimpleNamespace(points=b, uv=np.zeros((len(b),2)))
        m = np.column_stack([np.arange(len(a))]*2)
        initial, mask = verify_teaser_geometry(q, c, m)
        self.assertEqual(mask.sum(), len(a))
        A = cv2.Rodrigues(np.array([1.1, .7, -.4]))[0]
        B = cv2.Rodrigues(np.array([0., np.deg2rad(137), 0.]))[0]
        q.points = a @ A.T; c.points = b @ B.T
        q.uv[:] = np.nan; c.uv[:] = 1e9  # A sphere estimator never consumes these.
        result, changed = verify_teaser_geometry(q, c, m)
        np.testing.assert_array_equal(mask, changed)
        np.testing.assert_allclose(result['rotation'], B @ R @ A.T, atol=1e-7)
        np.testing.assert_allclose(result['translation'], B @ t, atol=1e-7)
        self.assertAlmostEqual(initial['scale'], result['scale'])

    def test_empty_invalid_depth_and_bad_inputs(self):
        a, b, _, _ = self.fixture()
        q, c = SimpleNamespace(points=a), SimpleNamespace(points=b)
        self.assertEqual(verify_teaser_geometry(q,c,np.empty((0,2),np.int32))[0]['verification_status'],'no_matches')
        a[1] = np.nan; a[2] = np.inf; b[3] = 0
        m = np.column_stack([np.arange(8)]*2)
        result, mask = verify_teaser_geometry(q,c,m)
        self.assertEqual(result['depth_matches'], 5)
        self.assertFalse(mask.any())
        for x,y,bound in [(a,b,.01),(b[:2],b[:2],.01),(b,b,0),(b,b,np.nan),(b[:,:2],b,.01)]:
            with self.assertRaises(ValueError): estimate_sim3(x,y,noise_bound=bound)
        with self.assertRaises(ValueError): verify_teaser_geometry(q,c,np.array([[-1,0]]))
        with self.assertRaises(ValueError): estimate_sim3(np.ones((8,3)),b[:8],noise_bound=.01)

    def test_coincident_locations_preserve_index_mapping(self):
        a,b,_,_ = self.fixture()
        a[7] = a[2]; b[7] = b[2]
        got = estimate_sim3(a,b,noise_bound=.01)
        self.assertEqual(got['coincident_locations_omitted'],1)
        self.assertNotIn(7,got['solver_input_indices'])
        self.assertTrue(set(got['clique_indices']).issubset(got['solver_input_indices']))
        self.assertAlmostEqual(got['scale'],1.8,places=7)


if __name__ == '__main__':
    unittest.main()

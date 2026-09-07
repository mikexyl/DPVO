import unittest

import cv2
import numpy as np

from dpvo.sphere_bow import match_features
from dpvo.sphere_match_gates import GATES, match_variants, unique_targets
from test_sphere_bow import features


def rays(longitudes):
    a = np.deg2rad(longitudes)
    return np.column_stack((np.sin(a), np.zeros(len(a)), np.cos(a)))


def nearby_fixture():
    q = np.zeros((3,32),np.uint8);q[0,0]=7;q[1,0]=1;q[2]=255
    c = np.zeros((3,32),np.uint8);c[1,0]=8;c[2]=255
    return features(rays([179.8,-179.8,90]),q), features(rays([10,10.2,90]),c)


class SphereMatchGateTests(unittest.TestCase):
    def test_strict_baseline_equivalence(self):
        rng=np.random.default_rng(31)
        desc=rng.integers(0,256,(120,32),np.uint8)
        desc[1]=desc[0]  # Exact descriptor ties must preserve the baseline.
        q=features(rng.normal(size=(120,3)),desc)
        c=features(rng.normal(size=(120,3)),desc.copy())
        c.descriptors[50:]=rng.integers(0,256,(70,32),np.uint8)
        variants=match_variants(q,c)
        np.testing.assert_array_equal(variants['strict'],match_features(q,c))
        for gate in GATES:
            np.testing.assert_array_equal(variants[gate],match_features(q,c,gate=gate))

    def test_patch_competition_cycle_and_rotations(self):
        q,c=nearby_fixture()
        initial=match_variants(q,c)
        self.assertNotIn((0,0),map(tuple,initial['forward-ratio']))  # 3/4 fails the strict ratio.
        self.assertIn((0,0),map(tuple,initial['patch-ratio-mutual']))
        self.assertNotIn((0,0),map(tuple,initial['strict']))
        self.assertNotIn((0,0),map(tuple,match_variants(q,c,patch_radius_degrees=.1)['patch-ratio-mutual']))
        for angle in ([np.pi/2,0,0],[0,np.pi,0],[.4,.7,1.1]):
            a,b=nearby_fixture()
            R=cv2.Rodrigues(np.asarray(angle,float))[0]
            a.bearings=a.bearings@R.T  # Independent sphere rotations, including seam -> pole.
            b.bearings=b.bearings@R
            a.points=np.full_like(a.points,np.nan);b.points=np.full_like(b.points,np.nan)
            rotated=match_variants(a,b)
            for gate in GATES:np.testing.assert_array_equal(initial[gate],rotated[gate])
        c.bearings=rays([10,10.2,10.1])  # No distinct competitor: conservatively reject.
        self.assertEqual(len(match_variants(q,c)['patch-ratio-mutual']),0)

    def test_unique_targets_empty_and_invalid(self):
        q,c=nearby_fixture()
        pairs=np.array([[0,0],[1,0],[2,2]],np.int32)
        np.testing.assert_array_equal(unique_targets(q,c,pairs),[[1,0],[2,2]])
        variants=match_variants(features(np.empty((0,3))),c)
        for result in variants.values():self.assertEqual(result.shape,(0,2))
        for radius in (0,-1,181,np.nan):
            with self.assertRaises(ValueError):match_variants(q,c,patch_radius_degrees=radius)
        with self.assertRaises(ValueError):match_features(q,c,gate='typo')
        c.descriptor_family='sphorb'
        with self.assertRaises(ValueError):match_variants(q,c)


if __name__=='__main__':unittest.main()

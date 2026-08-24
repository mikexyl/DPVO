import importlib.util
import types
import unittest
from unittest import mock

import numpy as np

from dpvo.loop_closure.distributed import (
    BowCandidateDetector,
    FrameIdentity,
    RobustSim3Verifier,
    SparseBow,
    bow_score,
)


class BowCandidateDetectorTest(unittest.TestCase):
    def test_dbow_l1_score_is_sparse_histogram_intersection(self):
        first = SparseBow([1, 3, 8], [0.2, 0.5, 0.3])
        second = SparseBow([1, 2, 8], [0.1, 0.7, 0.2])
        self.assertAlmostEqual(bow_score(first, second), 0.3, places=6)

    def test_requests_detail_only_after_repeated_remote_matches(self):
        detector = BowCandidateDetector(threshold=0.4, repetitions=3, nms_radius=50)
        detector.add_local(20, SparseBow([1, 2], [0.5, 0.5]))
        detector.add_local(80, SparseBow([3, 4], [0.5, 0.5]))
        remote = SparseBow([1, 2], [0.5, 0.5])

        self.assertIsNone(detector.observe(FrameIdentity("robot1", "run", 100), remote))
        self.assertIsNone(detector.observe(FrameIdentity("robot1", "run", 101), remote))
        match = detector.observe(FrameIdentity("robot1", "run", 102), remote)

        self.assertIsNotNone(match)
        self.assertEqual(match.local_keyframe_id, 20)
        self.assertEqual(match.remote_frame.keyframe_id, 101)
        detector.confirm(match)

        for keyframe_id in (103, 104, 105):
            repeated = detector.observe(
                FrameIdentity("robot1", "run", keyframe_id),
                remote,
            )
        self.assertIsNone(repeated, "NMS should suppress a nearby confirmed loop")


class RobustSim3VerifierTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.src = rng.normal(size=(80, 3))
        angle = 0.25
        self.rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.translation = np.array([0.4, -0.2, 0.8])
        self.scale = 1.3
        self.dst = self.scale * (self.src @ self.rotation.T) + self.translation

    def test_uses_teaser_when_bindings_are_available(self):
        expected_rotation = self.rotation
        expected_translation = self.translation
        expected_scale = self.scale

        class FakeSolver:
            class Params:
                pass

            class ROTATION_ESTIMATION_ALGORITHM:
                GNC_TLS = 1

            def __init__(self, params):
                self.params = params

            def solve(self, src, dst):
                self.src_is_fortran = src.flags.f_contiguous
                self.dst_is_fortran = dst.flags.f_contiguous
                if not self.src_is_fortran or not self.dst_is_fortran:
                    raise AssertionError("TEASER inputs must be column-major")

            def getSolution(self):
                return types.SimpleNamespace(
                    valid=True,
                    rotation=expected_rotation,
                    translation=expected_translation,
                    scale=expected_scale,
                )

        module = types.SimpleNamespace(RobustRegistrationSolver=FakeSolver)
        verifier = RobustSim3Verifier(0.05, min_inliers=30, min_inlier_ratio=0.5)
        with mock.patch(
            "dpvo.loop_closure.distributed.importlib.import_module",
            return_value=module,
        ):
            result = verifier.verify(self.src, self.dst)

        self.assertTrue(result.success)
        self.assertEqual(result.method, "teaser++")
        self.assertEqual(result.inliers, self.src.shape[0])

    def test_ransac_fallback_rejects_outliers(self):
        rng = np.random.default_rng(8)
        contaminated = self.dst.copy()
        contaminated[:25] = rng.normal(loc=20.0, scale=5.0, size=(25, 3))
        verifier = RobustSim3Verifier(0.05, min_inliers=40, min_inlier_ratio=0.5)
        with mock.patch(
            "dpvo.loop_closure.distributed.importlib.import_module",
            side_effect=ImportError,
        ):
            result = verifier.verify(self.src, contaminated)

        self.assertTrue(result.success)
        self.assertEqual(result.method, "ransac")
        self.assertGreaterEqual(result.inliers, 55)
        np.testing.assert_allclose(result.rotation, self.rotation, atol=1e-6)
        np.testing.assert_allclose(result.translation, self.translation, atol=1e-6)
        self.assertAlmostEqual(result.scale, self.scale, places=6)

    @unittest.skipUnless(
        importlib.util.find_spec("teaserpp_python"),
        "TEASER++ bindings are not installed",
    )
    def test_installed_teaser_binding(self):
        contaminated = self.dst.copy()
        contaminated[:20] += 30.0
        verifier = RobustSim3Verifier(
            0.05,
            min_inliers=50,
            min_inlier_ratio=0.5,
            teaser_required=True,
        )
        result = verifier.verify(self.src, contaminated)

        self.assertTrue(result.success)
        self.assertEqual(result.method, "teaser++")
        self.assertGreaterEqual(result.inliers, 60)


if __name__ == "__main__":
    unittest.main()

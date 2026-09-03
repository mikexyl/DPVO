import importlib.util
import json
import types
import unittest
from unittest import mock

import numpy as np

from dpvo.loop_closure.distributed import (
    BowCandidateDetector,
    FrameIdentity,
    GlobalDescriptor,
    GlobalDescriptorCandidateDetector,
    RobustSim3Verifier,
    SparseBow,
    bow_score,
    depth_statistics,
    valid_depth_mask,
)


class GlobalDescriptorCandidateDetectorTest(unittest.TestCase):
    @staticmethod
    def descriptor(values):
        return GlobalDescriptor(values, "gmberton/MegaLoc")

    def test_descriptor_is_validated_and_l2_normalized(self):
        descriptor = self.descriptor([3.0, 4.0])
        np.testing.assert_allclose(descriptor.values, [0.6, 0.8])
        with self.assertRaises(ValueError):
            self.descriptor([0.0, 0.0])
        with self.assertRaises(ValueError):
            self.descriptor([1.0, np.nan])

    def test_cosine_top1_requests_only_after_repeated_matches(self):
        detector = GlobalDescriptorCandidateDetector(
            threshold=0.8,
            repetitions=2,
            nms_radius=10,
        )
        detector.add_local(10, self.descriptor([1.0, 0.0, 0.0]))
        detector.add_local(50, self.descriptor([0.0, 1.0, 0.0]))
        remote = self.descriptor([0.99, 0.05, 0.0])

        self.assertIsNone(
            detector.observe(FrameIdentity("robot1", "run", 100), remote)
        )
        match = detector.observe(FrameIdentity("robot1", "run", 101), remote)

        self.assertIsNotNone(match)
        self.assertEqual(match.local_keyframe_id, 10)
        self.assertEqual(match.remote_frame.keyframe_id, 101)
        self.assertGreater(match.score, 0.99)

    def test_dense_retrieval_backfills_when_remote_arrives_first(self):
        detector = GlobalDescriptorCandidateDetector(
            threshold=0.9,
            repetitions=1,
            nms_radius=0,
        )
        remote = self.descriptor([0.0, 1.0, 0.0])
        self.assertIsNone(
            detector.observe(FrameIdentity("robot2", "run", 80), remote)
        )

        matches = detector.add_local(
            25,
            self.descriptor([0.0, 0.99, 0.01]),
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].local_keyframe_id, 25)
        self.assertEqual(matches[0].remote_frame.keyframe_id, 80)


class BowCandidateDetectorTest(unittest.TestCase):
    def test_dbow_l1_score_is_sparse_histogram_intersection(self):
        first = SparseBow([1, 3, 8], [0.2, 0.5, 0.3])
        second = SparseBow([1, 2, 8], [0.1, 0.7, 0.2])
        self.assertAlmostEqual(bow_score(first, second), 0.3, places=6)

    def test_inverted_index_matches_exhaustive_top1_scoring(self):
        rng = np.random.default_rng(17)
        detector = BowCandidateDetector(threshold=0.0, repetitions=1, nms_radius=0)
        local_bows = {}
        for keyframe_id in range(40):
            word_ids = np.sort(rng.choice(600, size=80, replace=False))
            values = rng.random(80)
            values /= values.sum()
            local_bows[keyframe_id] = SparseBow(word_ids, values)
            detector.add_local(keyframe_id, local_bows[keyframe_id])

        remote_ids = np.sort(rng.choice(600, size=90, replace=False))
        remote_values = rng.random(90)
        remote_values /= remote_values.sum()
        remote = SparseBow(remote_ids, remote_values)

        expected_id, expected_score = max(
            (
                (keyframe_id, bow_score(local_bow, remote))
                for keyframe_id, local_bow in local_bows.items()
            ),
            key=lambda item: item[1],
        )
        actual_id, actual_score = detector._best_local_match(remote)
        self.assertEqual(actual_id, expected_id)
        self.assertAlmostEqual(actual_score, expected_score, places=7)

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

    def test_nms_reserves_candidate_while_verification_is_pending(self):
        detector = BowCandidateDetector(threshold=0.4, repetitions=2, nms_radius=10)
        detector.add_local(20, SparseBow([1, 2], [0.5, 0.5]))
        remote = SparseBow([1, 2], [0.5, 0.5])

        self.assertIsNone(detector.observe(FrameIdentity("robot1", "run", 100), remote))
        pending = detector.observe(FrameIdentity("robot1", "run", 101), remote)
        self.assertIsNotNone(pending)

        repeated = detector.observe(FrameIdentity("robot1", "run", 102), remote)
        self.assertIsNone(repeated, "NMS should suppress a nearby pending loop")

        detector.reject(pending)
        self.assertIsNone(detector.observe(FrameIdentity("robot1", "run", 103), remote))
        retried = detector.observe(FrameIdentity("robot1", "run", 104), remote)
        self.assertIsNotNone(retried, "A failed verification must release its NMS slot")

    def test_historical_scheduler_allows_pending_neighbor_burst(self):
        detector = BowCandidateDetector(
            threshold=0.4,
            repetitions=2,
            nms_radius=10,
            reserve_inflight=False,
        )
        detector.add_local(20, SparseBow([1, 2], [0.5, 0.5]))
        remote = SparseBow([1, 2], [0.5, 0.5])

        self.assertIsNone(detector.observe(FrameIdentity("robot1", "run", 100), remote))
        pending = detector.observe(FrameIdentity("robot1", "run", 101), remote)
        self.assertIsNotNone(pending)
        repeated = detector.observe(FrameIdentity("robot1", "run", 102), remote)
        self.assertIsNotNone(
            repeated,
            "The historical scheduler queued neighbors before verification completed",
        )

        detector.reject(pending)
        self.assertIsNotNone(
            detector.observe(FrameIdentity("robot1", "run", 103), remote)
        )

    def test_backfills_remote_bows_when_local_robot_lags(self):
        detector = BowCandidateDetector(threshold=0.4, repetitions=2, nms_radius=10)
        remote = SparseBow([1, 2], [0.5, 0.5])

        self.assertIsNone(detector.observe(FrameIdentity("robot2", "run", 100), remote))
        self.assertIsNone(detector.observe(FrameIdentity("robot2", "run", 101), remote))

        matches = detector.add_local(20, SparseBow([1, 2], [0.5, 0.5]))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].local_keyframe_id, 20)
        self.assertEqual(matches[0].remote_frame.keyframe_id, 101)
        self.assertEqual(detector.backfill_candidates, 1)

    def test_backfill_updates_only_when_new_local_bow_is_the_top1_match(self):
        detector = BowCandidateDetector(threshold=0.4, repetitions=1, nms_radius=0)
        remote = SparseBow([1, 2], [0.5, 0.5])
        detector.add_local(10, SparseBow([1, 2], [0.5, 0.5]))
        online = detector.observe(FrameIdentity("robot2", "run", 100), remote)
        self.assertIsNotNone(online)
        detector.reject(online)

        weaker = detector.add_local(20, SparseBow([1, 2], [0.25, 0.25]))
        self.assertEqual(weaker, [])
        self.assertEqual(detector.remote_best[("robot2", "run")][100].local_keyframe_id, 10)

    def test_diagnostics_report_top1_scores_and_candidate_gates_by_remote(self):
        detector = BowCandidateDetector(threshold=0.4, repetitions=2, nms_radius=10)
        detector.add_local(20, SparseBow([1, 2], [0.5, 0.5]))
        remote_key = ("robot1", "run")

        detector.observe(
            FrameIdentity(*remote_key, 100),
            SparseBow([1, 3], [0.2, 0.8]),
        )
        detector.observe(
            FrameIdentity(*remote_key, 101),
            SparseBow([1, 2], [0.5, 0.5]),
        )
        match = detector.observe(
            FrameIdentity(*remote_key, 102),
            SparseBow([1, 2], [0.5, 0.5]),
        )

        self.assertIsNotNone(match)
        snapshot = detector.diagnostics_snapshot()
        remote = snapshot["by_remote"]["robot1:run"]
        self.assertEqual(snapshot["observations"], 3)
        self.assertEqual(snapshot["below_threshold_observations"], 1)
        self.assertEqual(snapshot["repetition_waits"], 1)
        self.assertEqual(remote["threshold_hits"], 2)
        self.assertEqual(remote["candidates"], 1)
        self.assertEqual(remote["top1_score"]["count"], 3)
        json.dumps(snapshot)

    def test_non_positive_max_depth_disables_scale_dependent_upper_bound(self):
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 30.0],
                [0.0, 0.0, -1.0],
                [np.nan, 0.0, 2.0],
            ]
        )
        np.testing.assert_array_equal(
            valid_depth_mask(points, 20.0),
            [True, False, False, False],
        )
        np.testing.assert_array_equal(
            valid_depth_mask(points, 0.0),
            [True, True, False, False],
        )
        stats = depth_statistics(points, 20.0)
        self.assertEqual(stats["positive_depth"], 2)
        self.assertEqual(stats["valid_depth"], 1)


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

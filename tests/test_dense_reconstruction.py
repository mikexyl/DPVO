import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from dpvo.loop_closure.centralized import Sim3
from dpvo.loop_closure.dense_reconstruction import (
    CACHE_FORMAT_NAME,
    CACHE_FORMAT_VERSION,
    INFERENCE_MODE,
    RAW_DPVO_INFERENCE_MODE,
    CbsPose,
    FrameKey,
    FusionSettings,
    InferenceSettings,
    PairJob,
    PairPrediction,
    InvalidDa3BaselineError,
    backproject_pixels,
    build_two_view_jobs,
    cbs_sim3_to_rigid_w2c,
    camera_centers_from_world_to_camera,
    concatenate_batches,
    dense_rrd_recording_id,
    filter_prediction_view,
    fuse_cached_jobs,
    infer_jobs,
    load_cbs_csv,
    load_pair_cache,
    pair_cache_path,
    pair_provenance,
    pair_scale_diagnostics,
    recover_dpvo_patch_observations,
    recover_full_resolution_intrinsics,
    raw_dpvo_poses_from_tracking,
    refine_prediction_scale_from_dpvo_keypoints,
    reprojection_consistency_mask,
    save_pair_cache,
    scale_prediction_to_cbs,
    validate_inputs,
    validate_raw_dpvo_inputs,
    voxel_fuse,
)
from dpvo.loop_closure.pose_graph import (
    PoseGraphEdge,
    PoseGraphVertex,
    Sim3PoseGraph,
)


def _pose(key, centre, scale=1.0, vertex_id=0, rotation=None):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    sim3 = Sim3(centre, rotation, scale)
    rigid = cbs_sim3_to_rigid_w2c(sim3)
    return CbsPose(
        vertex_id,
        key,
        rigid,
        np.asarray(centre, dtype=np.float64),
        rigid[:3, :3].T,
        scale,
    )


def _artifact(robot_id, count, root=None):
    root = Path(root or "/tmp")
    session_id = f"{robot_id}_session"

    class Artifact:
        keyframe_count = count
        keyframe_poses_xyzw = np.asarray(
            [[float(index), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for index in range(count)],
            dtype=np.float32,
        )
        internal_intrinsics = np.tile(
            np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32), (count, 1)
        )
        dpvo_resolution = 2

        def frame_path(self, keyframe_id):
            return root / f"{robot_id}_{keyframe_id}.jpg"

    artifact = Artifact()
    artifact.robot_id = robot_id
    artifact.session_id = session_id
    artifact.manifest = {
        "state_sha256": f"state-{robot_id}",
        "map_sha256": f"map-{robot_id}",
        "keyframe_images_sha256": f"images-{robot_id}",
    }
    return artifact


def _graph(vertices, edges=()):
    return Sim3PoseGraph("test", list(vertices), list(edges))


class GeometryTest(unittest.TestCase):
    def test_dense_rrd_recording_id_is_stable_and_output_specific(self):
        first = dense_rrd_recording_id(Path("/tmp/run_a/dense/map.rrd"))
        self.assertEqual(first, dense_rrd_recording_id(Path("/tmp/run_a/dense/map.rrd")))
        self.assertNotEqual(first, dense_rrd_recording_id(Path("/tmp/run_b/dense/map.rrd")))
        self.assertTrue(first.startswith("dpvo-cbs-da3-dense-map-"))

    def test_intrinsics_recovery(self):
        recovered = recover_full_resolution_intrinsics(
            np.asarray([90.0, 91.0, 120.0, 67.5]), 4
        )
        np.testing.assert_allclose(
            recovered,
            [[360.0, 0.0, 480.0], [0.0, 364.0, 270.0], [0.0, 0.0, 1.0]],
        )

    def test_cbs_sim3_to_rigid_w2c_ignores_scale(self):
        rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        w2c = cbs_sim3_to_rigid_w2c(Sim3([1.0, 2.0, 3.0], rotation, 7.0))
        np.testing.assert_allclose(w2c[:3, :3], rotation.T, atol=1e-12)
        np.testing.assert_allclose(w2c[:3, 3], -rotation.T @ [1.0, 2.0, 3.0])
        np.testing.assert_allclose(w2c[:3, :3].T @ w2c[:3, :3], np.eye(3))

    def test_cbs_sim3_rotation_is_projected_to_proper_so3(self):
        rotation = Rotation.from_euler("xyz", [10, -5, 20], degrees=True).as_matrix()
        contaminated = 2.5 * rotation
        w2c = cbs_sim3_to_rigid_w2c(Sim3([4.0, -2.0, 1.0], contaminated, 9.0))
        np.testing.assert_allclose(w2c[:3, :3], rotation.T, atol=1e-12)
        np.testing.assert_allclose(-w2c[:3, :3].T @ w2c[:3, 3], [4.0, -2.0, 1.0])
        self.assertAlmostEqual(float(np.linalg.det(w2c[:3, :3])), 1.0)

    def test_da3_world_to_camera_extrinsic_convention(self):
        rotation = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        centres = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
        w2c = np.tile(np.eye(4), (2, 1, 1))
        w2c[:, :3, :3] = rotation.T
        w2c[:, :3, 3] = -np.einsum("nij,nj->ni", w2c[:, :3, :3], centres)
        np.testing.assert_allclose(camera_centers_from_world_to_camera(w2c[:, :3]), centres)

    def test_pair_scale_recovery_and_scaled_backprojection(self):
        keys = (FrameKey("robot0", "s", 0), FrameKey("robot0", "s", 1))
        cbs_pair = [_pose(keys[0], [10, 0, 0]), _pose(keys[1], [20, 0, 0])]
        da3_w2c = np.tile(np.eye(4), (2, 1, 1))
        da3_w2c[1, 0, 3] = -2.0
        raw = PairPrediction(
            np.full((2, 1, 1), 2.0, dtype=np.float32),
            np.ones((2, 1, 1), dtype=np.float32),
            np.zeros((2, 1, 1, 3), dtype=np.uint8),
            np.tile(np.eye(3), (2, 1, 1)),
            da3_w2c,
        )
        scaled, diagnostics = scale_prediction_to_cbs(raw, cbs_pair)
        self.assertEqual(diagnostics["cbs_baseline"], 10.0)
        self.assertEqual(diagnostics["da3_baseline"], 2.0)
        self.assertEqual(diagnostics["applied_depth_scale"], 5.0)
        np.testing.assert_allclose(scaled.depth, 10.0)
        point = backproject_pixels(
            np.asarray([0]), np.asarray([0]), scaled.depth[0, 0, 0:1],
            scaled.intrinsics[0], cbs_pair[0].camera_from_world,
        )
        np.testing.assert_allclose(point, [[10.0, 0.0, 10.0]])

    def test_zero_and_nonfinite_da3_baselines_are_rejected(self):
        keys = (FrameKey("robot0", "s", 0), FrameKey("robot0", "s", 1))
        cbs_pair = [_pose(keys[0], [0, 0, 0]), _pose(keys[1], [1, 0, 0])]
        zero = np.tile(np.eye(4), (2, 1, 1))
        with self.assertRaisesRegex(InvalidDa3BaselineError, "effectively zero") as caught:
            pair_scale_diagnostics(zero, cbs_pair)
        self.assertIn("at_or_below", caught.exception.diagnostics["invalid_da3_baseline_reason"])
        nonfinite = zero.copy()
        nonfinite[1, 0, 3] = np.nan
        with self.assertRaisesRegex(InvalidDa3BaselineError, "cannot recover") as caught:
            pair_scale_diagnostics(nonfinite, cbs_pair)
        self.assertIn("non-finite", caught.exception.diagnostics["invalid_da3_baseline_reason"])

    def test_backprojection_is_in_cbs_world_frame(self):
        ext = cbs_sim3_to_rigid_w2c(Sim3([10, 0, 0], np.eye(3), 3.0))
        points = backproject_pixels(
            np.asarray([1.0]),
            np.asarray([0.0]),
            np.asarray([2.0]),
            np.asarray([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]),
            ext,
        )
        np.testing.assert_allclose(points, [[11.0, 0.0, 2.0]])

    def test_dpvo_keypoint_depth_refines_pair_scale(self):
        robot_id = "robot0"
        session_id = "session"
        keys = (
            FrameKey(robot_id, session_id, 0),
            FrameKey(robot_id, session_id, 1),
        )
        camera_points = np.asarray(
            [
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        artifact = SimpleNamespace(
            robot_id=robot_id,
            keyframe_count=2,
            patches_per_keyframe=4,
            map_points=np.tile(camera_points, (2, 1)),
            internal_poses_xyzw=np.tile(
                np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (2, 1)
            ),
            internal_intrinsics=np.tile(
                np.asarray([1, 1, 0, 0], dtype=np.float32), (2, 1)
            ),
            session_from_map=np.diag([3.0, 3.0, 3.0, 1.0]),
            dpvo_resolution=1,
        )
        observations = recover_dpvo_patch_observations(artifact)
        np.testing.assert_allclose(observations.stable_depth, 6.0)
        prediction = PairPrediction(
            np.full((2, 3, 3), 4.0, dtype=np.float32),
            np.ones((2, 3, 3), dtype=np.float32),
            np.zeros((2, 3, 3, 3), dtype=np.uint8),
            np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        )
        job = PairJob("pair", "intra_robot", keys, (True, True))
        refined, diagnostics = refine_prediction_scale_from_dpvo_keypoints(
            prediction,
            job,
            {robot_id: artifact},
            {
                keys[0]: _pose(keys[0], [0, 0, 0], scale=2),
                keys[1]: _pose(keys[1], [1, 0, 0], scale=2),
            },
            {robot_id: observations},
            {"applied_depth_scale": 2.0},
            FusionSettings(
                depth_scale_refinement="dpvo_keypoints",
                dpvo_keypoint_min_matches_per_view=2,
            ),
        )
        np.testing.assert_allclose(refined.depth, 12.0)
        self.assertAlmostEqual(diagnostics["dpvo_keypoint_depth_correction"], 3.0)
        self.assertAlmostEqual(diagnostics["applied_depth_scale"], 6.0)
        self.assertEqual(diagnostics["dpvo_keypoint_alignment"]["matches"], 8)


class PairingTest(unittest.TestCase):
    def _make(self, counts):
        artifacts = {}
        poses = {}
        vertices = []
        vertex_id = 0
        for robot_index, count in enumerate(counts):
            robot_id = f"robot{robot_index}"
            artifact = _artifact(robot_id, count)
            artifacts[robot_id] = artifact
            for keyframe_id in range(count):
                key = FrameKey(robot_id, artifact.session_id, keyframe_id)
                poses[key] = _pose(key, [keyframe_id, robot_index * 10, 0], vertex_id=vertex_id)
                vertices.append(
                    PoseGraphVertex(
                        vertex_id,
                        robot_id,
                        artifact.session_id,
                        Sim3.identity(),
                        keyframe_id=keyframe_id,
                    )
                )
                vertex_id += 1
        return artifacts, poses, vertices

    def test_even_and_odd_non_overlapping_pairing(self):
        artifacts, poses, vertices = self._make([4, 5])
        jobs = build_two_view_jobs(artifacts, _graph(vertices), poses)
        self.assertEqual(len(jobs), 5)
        self.assertEqual(
            [tuple(view.keyframe_id for view in job.views) for job in jobs],
            [(0, 1), (2, 3), (0, 1), (2, 3), (3, 4)],
        )
        self.assertEqual(jobs[-1].contributing_views, (False, True))
        covered = {
            view
            for job in jobs
            for view, contributes in zip(job.views, job.contributing_views)
            if contributes
        }
        self.assertEqual(len(covered), 9)

    def test_half_rate_stride_two_overlap_pairing(self):
        artifacts, poses, vertices = self._make([4, 5])
        jobs = build_two_view_jobs(
            artifacts,
            _graph(vertices),
            poses,
            intra_pair_gap=2,
            intra_pair_step=2,
        )
        self.assertEqual(
            [tuple(view.keyframe_id for view in job.views) for job in jobs],
            [(0, 2), (0, 2), (2, 4)],
        )
        self.assertTrue(all(job.contributing_views == (True, True) for job in jobs))
        self.assertTrue(all("intra_gap2_step2" in job.job_id for job in jobs))

    def test_zero_baseline_fallback_preserves_coverage(self):
        artifacts, poses, vertices = self._make([4])
        first = FrameKey("robot0", "robot0_session", 0)
        second = FrameKey("robot0", "robot0_session", 1)
        poses[second] = _pose(second, poses[first].camera_center, vertex_id=1)
        jobs = build_two_view_jobs(
            artifacts, _graph(vertices), poses, minimum_baseline=1e-6
        )
        fallbacks = [job for job in jobs if "fallback" in job.kind]
        self.assertEqual(len(fallbacks), 2)
        self.assertTrue(all(job.contributing_views == (True, False) for job in fallbacks))
        self.assertTrue(all(job.views[1].keyframe_id in (2, 3) for job in fallbacks))
        covered = {
            view.keyframe_id
            for job in jobs
            for view, contributes in zip(job.views, job.contributing_views)
            if contributes
        }
        self.assertEqual(covered, {0, 1, 2, 3})

    def test_loop_inclusion_and_duplicate_suppression(self):
        artifacts, poses, vertices = self._make([2, 2])
        edge = PoseGraphEdge(
            20, 0, 2, Sim3.identity(), np.ones(7), "inter_robot_loop_closure"
        )
        duplicate = PoseGraphEdge(
            21, 2, 0, Sim3.identity(), np.ones(7), "inter_robot_loop_closure"
        )
        jobs = build_two_view_jobs(artifacts, _graph(vertices, [edge, duplicate]), poses)
        loops = [job for job in jobs if job.kind == "inter_robot_loop_closure"]
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0].source_edge_id, 20)
        self.assertEqual(loops[0].contributing_views, (True, True))


class InputValidationTest(unittest.TestCase):
    def _valid_inputs(self):
        artifact = _artifact("robot0", 2)
        keys = [
            FrameKey("robot0", artifact.session_id, keyframe_id)
            for keyframe_id in range(2)
        ]
        vertices = [
            PoseGraphVertex(
                index,
                "robot0",
                artifact.session_id,
                Sim3.identity(),
                keyframe_id=index,
            )
            for index in range(2)
        ]
        graph = _graph(vertices)
        graph.metadata = {
            "pipeline_stage": "geometric_verification",
            "input_contains_global_optimization": False,
            "inter_robot_loop_count": 0,
            "artifact_state_sha256": {"robot0": "state-robot0"},
            "artifact_keyframe_images_sha256": {"robot0": "images-robot0"},
        }
        poses = {
            key: _pose(key, [index, 0, 0], vertex_id=index)
            for index, key in enumerate(keys)
        }
        return {"robot0": artifact}, graph, poses

    def test_incomplete_cbs_coverage_is_rejected(self):
        artifacts, graph, poses = self._valid_inputs()
        poses.pop(next(reversed(poses)))
        with self.assertRaisesRegex(ValueError, "missing 1 exact graph keyframes"):
            validate_inputs(artifacts, graph, poses)

    def test_unverified_graph_is_rejected(self):
        artifacts, graph, poses = self._valid_inputs()
        graph.metadata["pipeline_stage"] = "tracking"
        with self.assertRaisesRegex(ValueError, "not the verified"):
            validate_inputs(artifacts, graph, poses)

    def test_malformed_cbs_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            path.write_text("robot_id,keyframe_id\nrobot0,0\n")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_cbs_csv(path)

    def test_raw_dpvo_poses_come_directly_from_unoptimized_tracking(self):
        artifact = _artifact("robot0", 2)
        vertices = [
            PoseGraphVertex(
                index,
                "robot0",
                artifact.session_id,
                Sim3.from_pose(artifact.keyframe_poses_xyzw[index]),
                keyframe_id=index,
            )
            for index in range(2)
        ]
        graph = _graph(vertices)
        graph.metadata = {
            "pipeline_stage": "tracking",
            "contains_inter_robot_constraints": False,
            "contains_global_optimization": False,
        }
        artifacts = {"robot0": artifact}
        validate_raw_dpvo_inputs(artifacts, graph)
        poses = raw_dpvo_poses_from_tracking(artifacts, graph)
        second = poses[FrameKey("robot0", artifact.session_id, 1)]
        np.testing.assert_allclose(second.camera_center, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(second.camera_from_world[:3, 3], [-1.0, 0.0, 0.0])
        self.assertEqual(second.scale, 1.0)

        graph.metadata["contains_inter_robot_constraints"] = True
        with self.assertRaisesRegex(ValueError, "inter-robot constraints"):
            validate_raw_dpvo_inputs(artifacts, graph)


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = _artifact("robot0", 2, self.root)
        for keyframe_id in range(2):
            self.artifact.frame_path(keyframe_id).write_bytes(
                f"image {keyframe_id}".encode()
            )
        self.artifacts = {"robot0": self.artifact}
        self.keys = tuple(
            FrameKey("robot0", self.artifact.session_id, keyframe_id)
            for keyframe_id in range(2)
        )
        self.poses = {
            key: _pose(key, [index, 0, 0], vertex_id=index)
            for index, key in enumerate(self.keys)
        }
        self.job = PairJob("pair", "intra_robot", self.keys, (True, True))
        depth = np.full((2, 3, 4), 2.0, dtype=np.float32)
        self.prediction = PairPrediction(
            depth,
            np.ones_like(depth),
            np.full((2, 3, 4, 3), [10, 20, 30], dtype=np.uint8),
            np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
            np.stack([self.poses[key].camera_from_world for key in self.keys]),
        )
        self.settings = InferenceSettings()
        self.diagnostics = pair_scale_diagnostics(
            self.prediction.extrinsics,
            [self.poses[key] for key in self.keys],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_atomic_cache_recovery_and_validation(self):
        provenance = pair_provenance(
            self.job, self.artifacts, self.poses, self.settings
        )
        path = pair_cache_path(self.root, self.job)
        path.with_name(f".{path.name}.tmp").write_bytes(b"interrupted")
        self.assertIsNone(load_pair_cache(path, provenance))
        save_pair_cache(path, self.prediction, self.job, provenance, self.diagnostics)
        loaded = load_pair_cache(path, provenance)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded[0].processed_rgb, self.prediction.processed_rgb)

    def test_cache_invalidates_on_image_pose_model_or_setting_change(self):
        provenance = pair_provenance(
            self.job, self.artifacts, self.poses, self.settings
        )
        path = pair_cache_path(self.root, self.job)
        save_pair_cache(path, self.prediction, self.job, provenance, self.diagnostics)

        self.artifact.frame_path(0).write_bytes(b"changed image")
        changed_image = pair_provenance(
            self.job, self.artifacts, self.poses, self.settings
        )
        self.assertIsNone(load_pair_cache(path, changed_image))
        self.artifact.frame_path(0).write_bytes(b"image 0")

        moved = dict(self.poses)
        moved[self.keys[0]] = _pose(self.keys[0], [0.1, 0, 0], vertex_id=0)
        changed_pose = pair_provenance(
            self.job, self.artifacts, moved, self.settings
        )
        self.assertIsNone(load_pair_cache(path, changed_pose))

        changed_model = pair_provenance(
            self.job,
            self.artifacts,
            self.poses,
            replace(self.settings, weights_revision="different"),
        )
        self.assertIsNone(load_pair_cache(path, changed_model))
        changed_process = pair_provenance(
            self.job,
            self.artifacts,
            self.poses,
            replace(self.settings, process_res=518),
        )
        self.assertIsNone(load_pair_cache(path, changed_process))

    def test_cache_provenance_requires_pose_estimated_baseline_scaled_mode(self):
        provenance = pair_provenance(
            self.job, self.artifacts, self.poses, self.settings
        )
        self.assertEqual(provenance["cache_format"], CACHE_FORMAT_NAME)
        self.assertEqual(provenance["cache_version"], CACHE_FORMAT_VERSION)
        self.assertEqual(provenance["inference"]["inference_mode"], INFERENCE_MODE)
        self.assertFalse(provenance["inference"]["input_extrinsics"])
        self.assertFalse(provenance["inference"]["align_to_input_ext_scale"])

        path = pair_cache_path(self.root, self.job)
        save_pair_cache(path, self.prediction, self.job, provenance, self.diagnostics)
        old_provenance = dict(provenance)
        old_provenance["cache_format"] = "dpvo_cbs_da3_pair_cache"
        old_provenance["cache_version"] = 1
        self.assertIsNone(load_pair_cache(path, old_provenance))

    def test_raw_dpvo_cache_provenance_is_distinct_and_loadable(self):
        raw_settings = replace(
            self.settings,
            inference_mode=RAW_DPVO_INFERENCE_MODE,
        )
        raw_provenance = pair_provenance(
            self.job, self.artifacts, self.poses, raw_settings
        )
        cbs_provenance = pair_provenance(
            self.job, self.artifacts, self.poses, self.settings
        )
        self.assertNotEqual(raw_provenance["fingerprint"], cbs_provenance["fingerprint"])
        self.assertEqual(
            raw_provenance["inference"]["inference_mode"],
            RAW_DPVO_INFERENCE_MODE,
        )
        self.assertIn("raw_dpvo_pose_scale", raw_provenance["views"][0])
        self.assertNotIn("cbs_vertex_scale_diagnostic", raw_provenance["views"][0])
        path = pair_cache_path(self.root, self.job)
        save_pair_cache(
            path, self.prediction, self.job, raw_provenance, self.diagnostics
        )
        self.assertIsNotNone(load_pair_cache(path, raw_provenance))
        self.assertIsNone(load_pair_cache(path, cbs_provenance))

    def test_corrupt_cache_is_not_resumed(self):
        path = pair_cache_path(self.root, self.job)
        path.write_bytes(b"not an npz")
        provenance = pair_provenance(
            self.job, self.artifacts, self.poses, self.settings
        )
        self.assertIsNone(load_pair_cache(path, provenance))

    def test_fake_backend_inference_resume_and_fusion(self):
        prediction = self.prediction

        class FakeBackend:
            def __init__(self):
                self.calls = 0

            @property
            def revisions(self):
                return {"fake": True}

            def infer(self, image_paths, intrinsics, settings):
                self.calls += 1
                predicted_poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
                predicted_poses[1, 0, 3] = -0.5
                self.assertions = (
                    len(image_paths) == 2,
                    intrinsics.shape == (2, 3, 3),
                    settings.process_res == 504,
                    settings.input_extrinsics is False,
                    settings.align_to_input_ext_scale is False,
                )
                output = PairPrediction(
                    prediction.depth,
                    prediction.confidence,
                    prediction.processed_rgb,
                    intrinsics,
                    predicted_poses,
                )
                return output

        backend = FakeBackend()
        cache_dir = self.root / "cache"
        status = infer_jobs(
            [self.job],
            self.artifacts,
            self.poses,
            cache_dir,
            self.settings,
            backend,
        )
        self.assertEqual(status["written"], 1)
        self.assertEqual(status["rejected"], 0)
        self.assertEqual(backend.calls, 1)
        cached = load_pair_cache(
            pair_cache_path(cache_dir, self.job),
            pair_provenance(self.job, self.artifacts, self.poses, self.settings),
        )
        self.assertIsNotNone(cached)
        np.testing.assert_allclose(cached[0].depth, prediction.depth * 2.0)
        self.assertEqual(cached[3]["applied_depth_scale"], 2.0)
        self.assertTrue(all(backend.assertions))
        resumed = infer_jobs(
            [self.job],
            self.artifacts,
            self.poses,
            cache_dir,
            self.settings,
            backend,
        )
        self.assertEqual(resumed["reused"], 1)
        self.assertEqual(backend.calls, 1)

        cloud, stats = fuse_cached_jobs(
            [self.job],
            self.artifacts,
            self.poses,
            cache_dir,
            self.settings,
            FusionSettings(
                confidence_percentile=0,
                far_depth_percentile=100,
                reprojection_relative_tolerance=1.0,
                pixel_stride=1,
                voxel_size=0.01,
            ),
        )
        self.assertGreater(len(cloud.points), 0)
        self.assertEqual(stats["robots"]["robot0"]["contributing_views"], 2)


class FilteringAndFusionTest(unittest.TestCase):
    def _prediction(self):
        height, width = 4, 5
        depth = np.full((2, height, width), 2.0, dtype=np.float32)
        confidence = np.ones_like(depth)
        rgb = np.empty((2, height, width, 3), dtype=np.uint8)
        rgb[0] = [255, 0, 0]
        rgb[1] = [0, 0, 255]
        intrinsics = np.tile(
            np.asarray([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]], dtype=np.float32),
            (2, 1, 1),
        )
        extrinsics = np.stack(
            [
                cbs_sim3_to_rigid_w2c(Sim3([0, 0, 0], np.eye(3))),
                cbs_sim3_to_rigid_w2c(Sim3([0.5, 0, 0], np.eye(3))),
            ]
        )
        return PairPrediction(depth, confidence, rgb, intrinsics, extrinsics)

    def test_symmetric_reprojection_accepts_planar_depth(self):
        prediction = self._prediction()
        settings = FusionSettings(
            confidence_percentile=0,
            far_depth_percentile=100,
            reprojection_relative_tolerance=0.01,
            pixel_stride=1,
            voxel_size=0.01,
        )
        first, first_stats = filter_prediction_view(
            prediction, 0, settings, 0, prediction.extrinsics
        )
        second, second_stats = filter_prediction_view(
            prediction, 1, settings, 1, prediction.extrinsics
        )
        self.assertGreater(len(first.points), 0)
        self.assertGreater(len(second.points), 0)
        self.assertGreater(first_stats["projected_inside_partner"], 0)
        self.assertGreater(second_stats["projected_inside_partner"], 0)
        self.assertTrue(np.all(first.colors == [255, 0, 0]))
        self.assertTrue(np.all(second.colors == [0, 0, 255]))
        self.assertAlmostEqual(float(first.points[0, 2]), 2.0)

    def test_reprojection_rejects_inconsistent_partner_depth(self):
        prediction = self._prediction()
        prediction.depth[1] = 8.0
        points = backproject_pixels(
            np.asarray([2]), np.asarray([2]), np.asarray([2.0]),
            prediction.intrinsics[0], prediction.extrinsics[0],
        )
        keep, inside = reprojection_consistency_mask(
            points,
            prediction.depth[1],
            prediction.intrinsics[1],
            prediction.extrinsics[1],
            0.10,
        )
        self.assertTrue(inside[0])
        self.assertFalse(keep[0])

    def test_strict_reprojection_requires_partner_overlap(self):
        prediction = self._prediction()
        permissive, permissive_stats = filter_prediction_view(
            prediction,
            0,
            FusionSettings(
                confidence_percentile=0,
                far_depth_percentile=100,
                reprojection_relative_tolerance=0.01,
                require_reprojection_overlap=False,
                pixel_stride=1,
                voxel_size=0.01,
            ),
            0,
            prediction.extrinsics,
        )
        strict, strict_stats = filter_prediction_view(
            prediction,
            0,
            FusionSettings(
                confidence_percentile=0,
                far_depth_percentile=100,
                reprojection_relative_tolerance=0.01,
                require_reprojection_overlap=True,
                pixel_stride=1,
                voxel_size=0.01,
            ),
            0,
            prediction.extrinsics,
        )
        self.assertGreater(strict_stats["reprojection_rejected_outside"], 0)
        self.assertEqual(
            strict_stats["after_reprojection"],
            strict_stats["reprojection_consistent_inside"],
        )
        self.assertLess(len(strict.points), len(permissive.points))
        self.assertEqual(permissive_stats["reprojection_rejected_outside"], 0)

    def test_confidence_far_depth_and_stride_filters(self):
        prediction = self._prediction()
        prediction.confidence[0] = np.arange(20, dtype=np.float32).reshape(4, 5)
        prediction.depth[0, 3, 4] = 100.0
        settings = FusionSettings(
            confidence_percentile=40,
            far_depth_percentile=99.5,
            reprojection_relative_tolerance=10.0,
            pixel_stride=2,
            voxel_size=0.01,
        )
        batch, stats = filter_prediction_view(
            prediction, 0, settings, 0, prediction.extrinsics
        )
        self.assertLess(stats["after_confidence"], stats["positive_finite_depth"])
        self.assertLess(stats["after_far_depth"], stats["after_confidence"])
        self.assertEqual(len(batch.points), stats["after_stride"])

    def test_confidence_weighted_voxel_fusion_and_ownership(self):
        from dpvo.loop_closure.dense_reconstruction import PointBatch

        batch = PointBatch(
            points=np.asarray([[0.001, 0, 0], [0.009, 0, 0]], dtype=np.float32),
            colors=np.asarray([[255, 0, 0], [0, 0, 255]], dtype=np.uint8),
            weights=np.asarray([1.0, 3.0], dtype=np.float32),
            robot_ids=np.asarray([0, 1], dtype=np.int32),
            observations=np.asarray([1, 2], dtype=np.int32),
        )
        fused = voxel_fuse(batch, 0.02)
        self.assertEqual(len(fused.points), 1)
        np.testing.assert_allclose(fused.points[0], [0.007, 0, 0], atol=1e-6)
        np.testing.assert_array_equal(fused.colors[0], [64, 0, 191])
        self.assertEqual(fused.robot_ids[0], 1)
        self.assertEqual(fused.observations[0], 3)
        self.assertEqual(len(concatenate_batches([fused]).points), 1)


if __name__ == "__main__":
    unittest.main()

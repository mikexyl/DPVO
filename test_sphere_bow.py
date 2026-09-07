import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dpvo.sphere_bow import (FACE_ROTATIONS, SphereFeatures, assign_words, candidate_mask,
                           cube_views, extract_orb, face_bearings, fit_idf, fit_vocabulary,
                           match_features, panorama_uv, rank_candidates, tfidf,
                           verify_depth_geometry, word_counts)
from sphere_place_recognition import downsample_sphere, parser, run


def features(points, descriptors=None):
    points = np.asarray(points, np.float32)
    n = len(points)
    return SphereFeatures(np.zeros((n, 32), np.uint8) if descriptors is None else descriptors,
                          np.zeros((n, 2), np.float32), points, points,
                          np.zeros(n, np.uint8), np.zeros((n, 2), np.float32), np.full(n, 31))


class SphereBowTests(unittest.TestCase):
    def test_downsample_color_mask_depth_and_pixel_centers(self):
        bgra = np.full((8, 16, 4), 255, np.uint8)
        bgra[..., :3] = 0  # Valid black must survive.
        bgra[2:4, 2:4, :3] = np.array([[0, 40], [80, 120]])[..., None]
        bgra[0, 0, 3] = 0
        depth = np.arange(1, 129, dtype=np.float32).reshape(8, 16)
        depth[4, 4], depth[4, 6], depth[4, 8] = np.nan, np.inf, -1
        image, radial = downsample_sphere(bgra, depth, 2)
        self.assertEqual(image.shape, (4, 8, 4))
        np.testing.assert_array_equal(image[1, 1], [60, 60, 60, 255])
        np.testing.assert_array_equal(image[1, 0], [0, 0, 0, 255])
        self.assertEqual(np.count_nonzero(image[..., 3] == 0), 1)
        self.assertTrue(np.isnan(radial[0, 0]))
        self.assertTrue(np.isnan(radial[2, 2:5]).all())
        self.assertEqual(radial[1, 1], depth[2, 2])
        bgra[0, 0, :3] = [10, 200, 90]
        changed, changed_depth = downsample_sphere(bgra, depth, 2)
        np.testing.assert_array_equal(image, changed)
        np.testing.assert_array_equal(radial, changed_depth)
        for factor in (0, -1, 3, 8, 1.5):
            with self.assertRaises(ValueError):
                downsample_sphere(bgra, depth, factor)
        unchanged = downsample_sphere(bgra, depth)
        self.assertIs(unchanged[0], bgra)
        self.assertIs(unchanged[1], depth)

    def test_face_axes_and_pixel_centers(self):
        expected = [[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0], [0, 1, 0]]
        for i, direction in enumerate(expected):
            np.testing.assert_allclose(face_bearings([[127.5, 127.5]], i, 256), [direction], atol=1e-7)
            np.testing.assert_allclose(FACE_ROTATIONS[i].T @ FACE_ROTATIONS[i], np.eye(3))
            self.assertAlmostEqual(np.linalg.det(FACE_ROTATIONS[i]), 1)
        np.testing.assert_allclose(panorama_uv([[0, 0, 1]], 1024, 512), [[511.5, 255.5]])

    def test_seam_wrap_and_invalid_support(self):
        bgra = np.zeros((128, 256, 4), np.uint8)
        bgra[:, :32] = bgra[:, -32:] = [50, 100, 150, 255]
        _, rear, support = list(cube_views(bgra, 128))[2]
        self.assertTrue(support[64, 64])
        np.testing.assert_array_equal(rear[64, 64], [50, 100, 150])
        bgra[..., 3] = 0
        self.assertFalse(any(mask.any() for _, _, mask in cube_views(bgra, 128)))

    def test_orb_ignores_transparent_texture_and_valid_black_is_observed(self):
        rng = np.random.default_rng(9)
        bgra = rng.integers(0, 256, (256, 512, 4), np.uint8)
        bgra[..., 3] = 0
        depth = np.ones((256, 512), np.float32)
        empty = extract_orb(bgra, depth, face_size=128)
        self.assertEqual(empty.descriptors.shape, (0, 32))
        bgra[..., 3] = 255
        textured = extract_orb(bgra, depth, face_size=128)
        self.assertGreater(len(textured.descriptors), 20)
        np.testing.assert_allclose(np.linalg.norm(textured.points, axis=1), 1, atol=1e-6)
        bgra[..., :3] = 0
        self.assertTrue(all(mask.all() for _, _, mask in cube_views(bgra, 128)))
        self.assertEqual(len(extract_orb(bgra, depth, face_size=128).descriptors), 0)

    def test_rotated_multiscale_patch_never_touches_unknown(self):
        rng = np.random.default_rng(3)
        bgra = rng.integers(0, 256, (256, 512, 4), np.uint8)
        bgra[..., 3] = 255
        bgra[70:120, 150:250, 3] = 0
        found = extract_orb(bgra, np.ones((256, 512), np.float32), face_size=192)
        for face, _, mask in cube_views(bgra, 192):
            distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2,
                                             cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
            keep = found.face_ids == face
            xy = np.rint(found.face_xy[keep]).astype(int)
            required = np.ceil((np.sqrt(2) / 2 + 4 / 31) * found.sizes[keep])
            self.assertTrue(np.all(distance[xy[:, 1], xy[:, 0]] >= required))

    def test_hamming_assignment_not_euclidean_byte_distance(self):
        centers = np.zeros((2, 32), np.uint8)
        centers[:, 0] = [128, 3]
        self.assertEqual(assign_words(np.zeros((1, 32), np.uint8), centers)[0], 0)
        self.assertEqual(len(assign_words(np.empty((0, 32), np.uint8), centers)), 0)

    def test_binary_training_deterministic_and_degenerate_input(self):
        rng = np.random.default_rng(7)
        docs = [rng.integers(0, 256, (80, 32), np.uint8) for _ in range(3)]
        a, b = fit_vocabulary(docs, words=16), fit_vocabulary(docs, words=16)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(word_counts(docs, a).shape, (3, 16))
        with self.assertRaises(ValueError):
            fit_vocabulary([np.empty((0, 32), np.uint8)])
        with self.assertRaises(ValueError):
            fit_vocabulary([np.zeros((4, 32), np.uint8)])

    def test_tfidf_empty_identical_and_disjoint_documents(self):
        counts = np.asarray([[1, 0], [1, 0], [0, 2], [0, 0]], np.float32)
        idf = fit_idf(counts[:2])
        saved_idf = idf.copy()
        vectors = tfidf(counts, idf)
        np.testing.assert_array_equal(idf, saved_idf)
        np.testing.assert_allclose((vectors @ vectors.T)[0], [1, 1, 0, 0])
        self.assertTrue(np.isfinite(vectors).all())

    def test_past_only_source_overlap_and_training_boundary_purge(self):
        snapshots = [dict(anchor_timestamp=t, source_timestamps=s) for t, s in
                     [(10, [1, 2]), (20, [2, 3]), (30, [3, 4]), (40, [5, 6]), (50, [6, 7])]]
        mask = candidate_mask(snapshots, query_start=2)
        self.assertFalse(mask[:3].any())  # sphere 30 still shares training image 3
        np.testing.assert_array_equal(mask[3], [True, True, True, False, False])
        np.testing.assert_array_equal(mask[4], [True, True, True, False, False])
        self.assertFalse(candidate_mask(snapshots, 2, min_anchor_gap=50).any())
        scores = np.eye(5) + .5
        scores[4, 3] = 100  # overlapping candidate must NEVER win, even with highest score
        rankings = rank_candidates(scores, mask, top_k=2)
        self.assertEqual(rankings[4], [0, 1])
        with self.assertRaises(ValueError):
            candidate_mask(snapshots[::-1])

    def test_mutual_ratio_matches_and_empty(self):
        rng = np.random.default_rng(4)
        desc = rng.integers(0, 256, (20, 32), np.uint8)
        f = features(np.ones((20, 3)), desc)
        np.testing.assert_array_equal(match_features(f, f), np.column_stack((np.arange(20), np.arange(20))))
        self.assertEqual(len(match_features(features(np.empty((0, 3))), f)), 0)

    def test_depth_sim3_outliers_and_degeneracy(self):
        rng = np.random.default_rng(9)
        points = rng.normal(size=(60, 3)) + [0, 0, 4]
        rotation = cv2.Rodrigues(np.asarray([.1, .2, -.3]))[0]
        target = 1.4 * points @ rotation.T + [.3, -.2, 1]
        target[-15:] = rng.normal(size=(15, 3))
        matches = np.column_stack((np.arange(60), np.arange(60)))
        result, mask = verify_depth_geometry(features(points), features(target), matches, iterations=100)
        self.assertGreaterEqual(result['geometric_inliers'], 45)
        self.assertTrue(mask[:45].all())
        self.assertAlmostEqual(result['scale'], 1.4, places=5)
        result, mask = verify_depth_geometry(features(np.ones((60, 3))), features(target), matches, iterations=10)
        self.assertEqual(result['geometric_inliers'], 0)

    def test_depth_opposite_heading_and_unverified_matches(self):
        from inspect_sphere_matches import depth_display_masks
        points = np.random.default_rng(35).normal(size=(60, 3))
        target = 1.4 * points @ np.diag([-1., 1., -1.]) + [.3, -.2, 1.]
        a, b = features(points), features(target)
        matches = np.column_stack((np.arange(60), np.arange(60)))
        result, mask = verify_depth_geometry(a, b, matches)
        self.assertEqual(result['verification_status'], 'checked')
        self.assertTrue(mask.all())
        result, mask = verify_depth_geometry(a, b, matches[:3])
        self.assertEqual(result['verification_status'], 'insufficient_depth_matches')
        rejected, unknown = depth_display_masks(a, b, matches[:3], result, mask)
        self.assertFalse(rejected.any())
        self.assertTrue(unknown.all())
        b.points[-1] = np.nan
        result, mask = verify_depth_geometry(a, b, matches)
        rejected, unknown = depth_display_masks(a, b, matches, result, mask)
        self.assertTrue(mask[:-1].all())
        self.assertFalse(rejected.any())
        np.testing.assert_array_equal(np.flatnonzero(unknown), [59])

    def test_crossing_panorama_matches_and_seam_invariance(self):
        longitude=np.deg2rad(np.linspace(-175,175,60))
        latitude=.5*np.sin(3*longitude)
        rays=np.column_stack((np.cos(latitude)*np.sin(longitude),np.sin(latitude),
                              np.cos(latitude)*np.cos(longitude)))
        points=rays*(4+.8*np.cos(2*longitude))[:,None]
        opposite=1.4*points@np.diag([-1.,1.,-1.])+[.3,-.2,1.]
        a,b=features(points),features(opposite)
        for f in (a,b):
            f.bearings=f.points/np.linalg.norm(f.points,axis=1)[:,None]
            f.uv=panorama_uv(f.bearings,1024,512)
        matches=np.column_stack((np.arange(60),np.arange(60)))
        dx=b.uv[:,0]-a.uv[:,0]
        self.assertGreater(np.count_nonzero(dx>0),20)
        self.assertGreater(np.count_nonzero(dx<0),20)
        geometry,mask=verify_depth_geometry(a,b,matches)
        self.assertTrue(mask.all())  # Both branches of the crossing panorama lines.
        seam_rotation=cv2.Rodrigues(np.array([0.,np.deg2rad(137),0.]))[0]
        b.points=b.points@seam_rotation.T
        b.bearings=b.bearings@seam_rotation.T
        b.uv=panorama_uv(b.bearings,1024,512)
        shifted,other=verify_depth_geometry(a,b,matches)
        np.testing.assert_array_equal(mask,other)
        self.assertAlmostEqual(geometry['scale'],shifted['scale'],places=6)

    def test_end_to_end_artifacts_vocabulary_reuse_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(6)
            bgra = rng.integers(0, 256, (128, 256, 4), np.uint8)
            bgra[..., 3] = 255
            snapshots = []
            for i in range(4):
                subdir = root / f'keyframe_{i:06d}'
                subdir.mkdir()
                cv2.imwrite(str(subdir / 'rgb.png'), bgra)
                np.savez_compressed(subdir / 'projection.npz', radial_depth=np.ones((128, 256), np.float32))
                snapshots.append(dict(anchor_timestamp=i, source_timestamps=[i], directory=subdir.name))
            (root / 'manifest.json').write_text(json.dumps(dict(snapshots=snapshots)))
            args = parser().parse_args(['--spheres', str(root), '--train-spheres', '2', '--words', '16',
                                        '--face-size', '128', '--features-per-face', '30', '--no-rerun'])
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report['summary']['eligible_queries'], 2)
            self.assertTrue((root / 'orb_bow' / 'report.json').exists())
            with self.assertRaises(FileExistsError):
                run(args)
            args.output = root / 'reused'
            args.vocabulary = root / 'orb_bow' / 'vocabulary.npz'
            with contextlib.redirect_stdout(io.StringIO()):
                reused = run(args)
            self.assertEqual(reused['summary']['eligible_queries'], 2)
            self.assertEqual(reused['queries'][0]['candidates'], report['queries'][0]['candidates'])


if __name__ == '__main__':
    unittest.main()

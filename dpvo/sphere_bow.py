"""CPU-only ORB / binary bag-of-words retrieval for saved local RGB spheres.

This is an experimental flat Hamming vocabulary, not the optional DBoW2 backend.
Retrieval uses only binary descriptors. The virtual-input sampler uses poses
and depth upstream to access source-image pixels; depth verification remains a
separate diagnostic from descriptor ranking.
"""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


FACE_NAMES = ("front", "right", "back", "left", "up", "down")
DESCRIPTOR_VERSIONS = {'cube-orb': 'opencv-orb-wta2-v1', 'sphorb': 'e5f2ccf-mask-rdf-v1',
                       'sphorb-virtual': 'e5f2ccf-source-grid-v1'}
# Columns are right, down, forward in the sphere's camera-RDF coordinates.
FACE_ROTATIONS = np.asarray([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
], dtype=np.float32)


def panorama_uv(bearings, width, height):
    """Continuous pixel-center coordinates; longitude wraps at the rear seam."""
    bearings = bearings / np.linalg.norm(bearings, axis=-1, keepdims=True)
    return np.stack(((np.arctan2(bearings[..., 0], bearings[..., 2]) / (2 * np.pi) + .5) * width - .5,
                     (np.arcsin(np.clip(bearings[..., 1], -1, 1)) / np.pi + .5) * height - .5), axis=-1)


def face_bearings(xy, face, size):
    rays = np.concatenate(((np.asarray(xy) + .5 - size / 2) / (size / 2),
                           np.ones((*np.asarray(xy).shape[:-1], 1))), axis=-1)
    rays = rays @ FACE_ROTATIONS[face].T
    return (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)


def _native_call(timing, operation, *args, **kwargs):
    if timing is None:
        return operation(*args, **kwargs)
    start = perf_counter()
    result = operation(*args, **kwargs)
    timing['native_seconds'] += perf_counter() - start
    return result


def cube_views(bgra, size=384, *, timing=None):
    """Six non-overlapping 90-degree views, with strictly observed RGB support."""
    if bgra.ndim != 3 or bgra.shape[2] != 4 or size < 96:
        raise ValueError("expected an RGBA/BGRA panorama and face size >= 96")
    height, width = bgra.shape[:2]
    y, x = np.mgrid[:size, :size]
    xy = np.stack((x, y), axis=-1)
    for face in range(6):
        uv = panorama_uv(face_bearings(xy, face, size), width, height).astype(np.float32)
        uv[..., 1] = np.clip(uv[..., 1], 0, height - 1)
        color = _native_call(timing, cv2.remap, bgra[..., :3], uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)
        support = _native_call(timing, cv2.remap, (bgra[..., 3] == 255).astype(np.float32), uv[..., 0], uv[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) > .999
        yield face, color, support


@dataclass
class SphereFeatures:
    descriptors: np.ndarray
    uv: np.ndarray
    bearings: np.ndarray
    points: np.ndarray
    face_ids: np.ndarray | None
    face_xy: np.ndarray | None
    sizes: np.ndarray
    responses: np.ndarray | None = None
    octaves: np.ndarray | None = None
    orientations: np.ndarray | None = None
    descriptor_family: str = 'cube-orb'
    descriptor_version: str = DESCRIPTOR_VERSIONS['cube-orb']

    def save(self, path):
        np.savez_compressed(path, **{k: v for k, v in vars(self).items() if v is not None})


def validate_descriptor_tag(metadata, family, version):
    """Untagged legacy vocabularies belong exclusively to cube ORB."""
    stored_family = str(np.asarray(metadata.get('descriptor_family', 'cube-orb')).item())
    stored_version = str(np.asarray(metadata.get('descriptor_version', DESCRIPTOR_VERSIONS['cube-orb'])).item())
    if (stored_family, stored_version) != (family, version):
        raise ValueError(f'incompatible descriptor vocabulary: {stored_family}/{stored_version}, expected {family}/{version}')


def extract_orb(bgra, radial_depth, face_size=384, features_per_face=500, *, timing=None):
    """ORB on perspective cube views, rejecting unknown pixels in each patch.

    The distance-to-invalid test grows with octave and includes the rotated
    descriptor footprint plus blur support. No inpainting of missing geometry.
    Features are mapped back to the original panorama and sphere bearings.
    """
    start = perf_counter()
    if timing is not None:
        timing.clear()
        timing['native_seconds'] = 0.
    if radial_depth.shape != bgra.shape[:2] or features_per_face < 1:
        raise ValueError("depth must match the panorama; feature budget must be positive")
    orb = _native_call(timing, cv2.ORB_create, nfeatures=features_per_face * 3, nlevels=6, fastThreshold=12,
                         edgeThreshold=31, patchSize=31, WTA_K=2)
    descriptors, uv_all, bearing_all, face_ids, face_xy, sizes = [], [], [], [], [], []
    responses, octaves, orientations = [], [], []
    for face, color, support in cube_views(bgra, face_size, timing=timing):
        distance = _native_call(timing, cv2.distanceTransform, np.pad(support.astype(np.uint8), 1), cv2.DIST_L2,
                                         cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        gray = _native_call(timing, cv2.cvtColor, color, cv2.COLOR_BGR2GRAY)
        keypoints = _native_call(timing, orb.detect, gray, (distance >= 27).astype(np.uint8) * 255)
        keypoints = [kp for kp in keypoints if distance[round(kp.pt[1]), round(kp.pt[0])] >=
                     np.ceil((np.sqrt(2) / 2 + 4 / 31) * kp.size)]
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:features_per_face]
        if not keypoints:
            continue
        keypoints, desc = _native_call(timing, orb.compute, gray, keypoints)
        xy = np.asarray([kp.pt for kp in keypoints], np.float32)
        bearings = face_bearings(xy, face, face_size)
        descriptors.append(desc)
        uv_all.append(panorama_uv(bearings, bgra.shape[1], bgra.shape[0]))
        bearing_all.append(bearings)
        face_ids.extend([face] * len(keypoints))
        face_xy.extend(xy)
        sizes.extend(kp.size for kp in keypoints)
        responses.extend(kp.response for kp in keypoints)
        octaves.extend(kp.octave for kp in keypoints)
        orientations.extend(kp.angle for kp in keypoints)
    desc = np.concatenate(descriptors) if descriptors else np.empty((0, 32), np.uint8)
    uv = np.concatenate(uv_all).astype(np.float32) if uv_all else np.empty((0, 2), np.float32)
    bearings = np.concatenate(bearing_all) if bearing_all else np.empty((0, 3), np.float32)
    x = np.floor(uv[:, 0] + .5).astype(int) % bgra.shape[1]
    y = np.clip(np.floor(uv[:, 1] + .5).astype(int), 0, bgra.shape[0] - 1)
    depth = radial_depth[y, x]
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    result = SphereFeatures(desc, uv, bearings, bearings * depth[:, None],
                          np.asarray(face_ids, np.uint8), np.asarray(face_xy, np.float32).reshape(-1, 2),
                          np.asarray(sizes, np.float32), np.asarray(responses, np.float32),
                          np.asarray(octaves, np.int32), np.asarray(orientations, np.float32))
    if timing is not None:
        timing['extraction_seconds'] = perf_counter() - start
        timing['wrapper_seconds'] = timing['extraction_seconds'] - timing['native_seconds']
    return result


class CubeOrbExtractor:
    """Instrumented baseline; ORB configuration and per-call construction are unchanged."""
    def __init__(self, face_size=384, features_per_face=500):
        start = perf_counter()
        self.face_size, self.features_per_face = face_size, features_per_face
        self.last_timing = {}
        self.setup_timing = dict(table_load_and_native_setup_seconds=0., total_setup_seconds=perf_counter() - start)

    def __call__(self, bgra, radial_depth):
        return extract_orb(bgra, radial_depth, self.face_size, self.features_per_face, timing=self.last_timing)


def assign_words(descriptors, centers):
    if not len(centers):
        raise ValueError("empty vocabulary")
    if not len(descriptors):
        return np.empty(0, np.int32)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).match(np.ascontiguousarray(descriptors),
                                                 np.ascontiguousarray(centers))
    words = np.empty(len(descriptors), np.int32)
    for match in matches:
        words[match.queryIdx] = match.trainIdx
    return words


def fit_vocabulary(documents, words=512, iterations=12, seed=7, max_descriptors=30000):
    """Binary k-medians: Hamming assignment and per-bit majority centers.

    Sampling is balanced per training sphere. The caller must use training
    documents only, and freeze both vocabulary and IDF before querying.
    """
    if words < 2 or iterations < 1 or max_descriptors < words:
        raise ValueError("need >= 2 words, >= 1 iteration and sufficient training capacity")
    rng = np.random.default_rng(seed)
    nonempty = [d for d in documents if len(d)]
    if not nonempty:
        raise ValueError("no ORB descriptors in the training spheres")
    per_document = max(1, max_descriptors // len(nonempty))
    sample = np.concatenate([d[rng.choice(len(d), min(len(d), per_document), replace=False)]
                             for d in nonempty])
    sample = np.unique(sample, axis=0)
    if len(sample) < 2:
        raise ValueError("need at least two distinct training descriptors")
    centers = sample[rng.choice(len(sample), min(words, len(sample)), replace=False)].copy()
    bits = np.unpackbits(sample, axis=1)
    for _ in range(iterations):
        labels = assign_words(sample, centers)
        updated = centers.copy()
        for word in range(len(centers)):
            group = bits[labels == word]
            if len(group):
                votes = group.sum(axis=0) * 2
                # Keep old bits on a tie, which also avoids arbitrary oscillation.
                old = np.unpackbits(centers[word])
                updated[word] = np.packbits(np.where(votes == len(group), old, votes > len(group)))
            else:
                updated[word] = sample[rng.integers(len(sample))]
        if np.array_equal(updated, centers):
            break
        centers = updated
    return np.unique(centers, axis=0)


def word_counts(documents, centers):
    return np.asarray([np.bincount(assign_words(d, centers), minlength=len(centers))
                       for d in documents], np.float32)


def fit_idf(counts):
    if len(counts) == 0:
        raise ValueError("IDF requires training documents")
    return (np.log((1 + len(counts)) / (1 + np.count_nonzero(counts, axis=0))) + 1).astype(np.float32)


def tfidf(counts, idf):
    weighted = np.where(counts > 0, 1 + np.log(np.maximum(counts, 1)), 0) * idf
    return weighted / np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12)


def candidate_mask(snapshots, query_start=0, min_anchor_gap=0):
    """Past-only, held-out queries; NEVER match overlapping source-keyframe sets."""
    if not 0 <= query_start <= len(snapshots) or min_anchor_gap < 0:
        raise ValueError("invalid query start or negative anchor gap")
    sources = [set(s['source_timestamps']) for s in snapshots]
    timestamps = [s['anchor_timestamp'] for s in snapshots]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("sphere anchors must be strictly chronological")
    allowed = np.zeros((len(snapshots), len(snapshots)), bool)
    training_sources = set().union(*sources[:query_start])
    for query in range(query_start, len(snapshots)):
        # Purge the train/query boundary too: a later sphere may still contain
        # images used to learn the vocabulary, despite having a new anchor.
        if not sources[query].isdisjoint(training_sources):
            continue
        for candidate in range(query):
            allowed[query, candidate] = (timestamps[query] - timestamps[candidate] >= min_anchor_gap
                                         and sources[query].isdisjoint(sources[candidate]))
    return allowed


def rank_candidates(similarity, allowed, top_k=3):
    if top_k < 1 or similarity.shape != allowed.shape:
        raise ValueError("positive top_k and matching score/mask shapes required")
    return [sorted(np.flatnonzero(mask), key=lambda j: (-float(row[j]), int(j)))[:top_k]
            for row, mask in zip(similarity, allowed)]


def match_features(query, candidate, ratio=.75, max_distance=64, gate='strict', patch_radius_degrees=1.):
    """Strict two-sided mutual matching by default; optional appearance gates."""
    if gate != 'strict':
        from .sphere_match_gates import GATES, match_variants
        if gate not in GATES:
            raise ValueError(f'unknown appearance gate: {gate}')
        return match_variants(query, candidate, ratio, max_distance, patch_radius_degrees)[gate]
    validate_descriptor_tag(vars(candidate), query.descriptor_family, query.descriptor_version)
    if len(query.descriptors) < 2 or len(candidate.descriptors) < 2:
        return np.empty((0, 2), np.int32)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def accepted(a, b):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 for m, n in [pair]
                if m.distance < ratio * n.distance and m.distance <= max_distance}

    forward = accepted(query.descriptors, candidate.descriptors)
    reverse = accepted(candidate.descriptors, query.descriptors)
    return np.asarray([(i, j) for i, j in forward.items() if reverse.get(j) == i], np.int32).reshape(-1, 2)


def _similarity_transform(source, target):
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    u, singular, vt = np.linalg.svd(b.T @ a / len(a))
    if singular[1] < 1e-10:
        raise ValueError("degenerate correspondence geometry")
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float((singular * sign).sum() / np.mean(np.sum(a * a, axis=1)))
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return scale, rotation, translation


def verify_depth_geometry(query, candidate, matches, seed=7, iterations=500, relative_threshold=.03):
    """Diagnostic Sim(3) RANSAC on DA3 radial depths; never modifies DPVO.

    A small residual is supporting evidence, not a calibrated loop-closure test:
    projected depth is noisy and office structures can alias. Monocular map
    scale can vary between saved snapshots, so fit scale as well as rigid pose.
    Rotation is unrestricted, including opposite headings. Unless the returned
    verification_status is 'checked', false inlier flags mean unverified, not
    rejected by a fitted model. Matches lacking finite depth stay unverified.
    """
    inliers = np.zeros(len(matches), bool)
    result = dict(depth_matches=0, geometric_inliers=0, geometric_inlier_ratio=0.0,
                  scale=None, median_relative_residual=None, verification_status='no_matches')
    if not len(matches):
        return result, inliers
    source, target = query.points[matches[:, 0]], candidate.points[matches[:, 1]]
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[valid].astype(np.float64), target[valid].astype(np.float64)
    result['depth_matches'] = len(source)
    if len(source) < 6:
        result['verification_status'] = 'insufficient_depth_matches'
        return result, inliers
    thresholds = np.maximum(np.linalg.norm(target, axis=1), 1e-6) * relative_threshold
    rng, best = np.random.default_rng(seed), np.zeros(len(source), bool)
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        try:
            scale, rotation, translation = _similarity_transform(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not .25 <= scale <= 4:
            continue
        residual = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
        consensus = residual < thresholds
        if consensus.sum() > best.sum():
            best = consensus
    if best.sum() < 6:
        result['verification_status'] = 'insufficient_consensus'
        return result, inliers
    try:
        scale, rotation, translation = _similarity_transform(source[best], target[best])
    except (ValueError, np.linalg.LinAlgError):
        result['verification_status'] = 'degenerate_refit'
        return result, inliers
    if not .25 <= scale <= 4:
        result['verification_status'] = 'scale_out_of_range'
        return result, inliers
    residual = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
    best = residual < thresholds
    inliers[np.flatnonzero(valid)] = best
    result.update(geometric_inliers=int(best.sum()), geometric_inlier_ratio=float(best.mean()),
                  verification_status='checked', scale=scale, median_relative_residual=float(np.median(
                      residual[best] / np.linalg.norm(target[best], axis=1))) if best.any() else None)
    return result, inliers

"""Optional appearance gates for saved spherical descriptors.

Patch cycles compare unit bearings within each sphere, never panorama pixels
or headings between cameras. No depth or pose enters these appearance gates.
"""
import cv2
import numpy as np

from .sphere_bow import validate_descriptor_tag

GATES = ('strict', 'forward-ratio', 'mutual-forward-ratio', 'patch-mutual',
         'patch-ratio-mutual', 'distance-only')


def match_variants(query, candidate, ratio=.75, max_distance=64, patch_radius_degrees=1.):
    """Return a fixed set of gate ablations from shared Hamming neighbors.

    Patch-ratio-mutual uses the nearest competitor outside the best match's
    angular patch, searched among the closest 32 descriptors. If none exists,
    it conservatively rejects the match. Relaxed gates can be many-to-one;
    use unique_targets for a separate, independent-target depth diagnostic.
    """
    validate_descriptor_tag(vars(candidate), query.descriptor_family, query.descriptor_version)
    if not 0 < ratio <= 1 or not 0 <= max_distance <= 256 or not 0 < patch_radius_degrees <= 180:
        raise ValueError('invalid descriptor ratio, distance or angular patch radius')
    empty = {name: np.empty((0, 2), np.int32) for name in GATES}
    if not len(query.descriptors) or not len(candidate.descriptors):
        return empty
    bearings = []
    for feature in (query, candidate):
        b = np.asarray(feature.bearings, np.float64)
        norm = np.linalg.norm(b, axis=1)
        if b.shape != (len(feature.descriptors), 3) or not np.isfinite(b).all() or (norm <= 0).any():
            raise ValueError('patch gates require finite nonzero bearings')
        bearings.append(b / norm[:, None])
    qb, cb = bearings
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(query.descriptors, candidate.descriptors, k=min(32, len(cb)))
    reverse = matcher.knnMatch(candidate.descriptors, query.descriptors, k=min(2, len(qb)))
    cosine = np.cos(np.deg2rad(patch_radius_degrees))
    result = {name: [] for name in GATES}
    for row in forward:
        m = row[0]
        if m.distance > max_distance:
            continue
        i, j = m.queryIdx, m.trainIdx
        pair = (i, j)
        result['distance-only'].append(pair)
        back = reverse[j]
        exact = back[0].trainIdx == i
        cycle = qb[i] @ qb[back[0].trainIdx] >= cosine - 1e-12
        forward_ratio = len(row) >= 2 and m.distance < ratio * row[1].distance
        if forward_ratio:
            result['forward-ratio'].append(pair)
            if exact:
                result['mutual-forward-ratio'].append(pair)
                if len(back) >= 2 and back[0].distance < ratio * back[1].distance and back[0].distance <= max_distance:
                    result['strict'].append(pair)
            if cycle:
                result['patch-mutual'].append(pair)
        if cycle:
            competitor = next((n for n in row[1:] if cb[j] @ cb[n.trainIdx] < cosine - 1e-12), None)
            if competitor is not None and m.distance < ratio * competitor.distance:
                result['patch-ratio-mutual'].append(pair)
    return {name: np.asarray(pairs, np.int32).reshape(-1, 2) for name, pairs in result.items()}


def unique_targets(query, candidate, matches):
    """Keep the lowest-Hamming match per candidate ID, then restore query order."""
    if not len(matches):
        return matches.copy()
    xor = np.bitwise_xor(query.descriptors[matches[:, 0]], candidate.descriptors[matches[:, 1]])
    distances = np.unpackbits(xor, axis=1).sum(axis=1)
    order = np.lexsort((matches[:, 0], distances))
    seen, kept = set(), []
    for k in order:
        target = int(matches[k, 1])
        if target not in seen:
            seen.add(target)
            kept.append(k)
    kept.sort(key=lambda k: int(matches[k, 0]))
    return matches[kept].copy()

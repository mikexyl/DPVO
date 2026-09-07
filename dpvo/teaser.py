"""Optional TEASER++ Sim(3) verification of spherical 3D correspondences.

No camera projection is used: points are RDF unit bearings times radial depth.
The native solver uses an absolute, isotropic noise bound. The final diagnostic
gate remains a separate per-point fraction of target radial range.
"""
from time import perf_counter

import numpy as np


def estimate_sim3(source, target, *, noise_bound, workers=4, clique_time_limit=2.):
    """Fit target = scale * R * source + t; noise_bound is in target units.

    A fresh native solver is used on every call. A common normalization of both
    clouds keeps numeric magnitudes reasonable without changing their geometry.
    No per-point normalization, projection, pose prior, or image gate is applied.
    """
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError('source and target must have equal shape (N, 3)')
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('at least three finite 3D correspondences are required')
    if not np.isfinite(noise_bound) or noise_bound <= 0:
        raise ValueError('noise_bound must be positive and finite, in target map units')
    if not isinstance(workers, (int, np.integer)) or not 1 <= workers <= 32:
        raise ValueError('workers must be an integer in [1, 32]')
    if not np.isfinite(clique_time_limit) or not 0 < clique_time_limit <= 60:
        raise ValueError('clique_time_limit must be in (0, 60] seconds')
    normalizer = float(np.median(np.linalg.norm(target, axis=1)))
    if not normalizer > 0:
        raise ValueError('target range must be positive')
    # Upstream TLS divides by source pair distances. Coincident detections across
    # octaves would create zero-length TIMs; keep the first exact location pair.
    seen_source, seen_target, kept = set(), set(), []
    for i, (a, b) in enumerate(zip(source, target)):
        ka, kb = tuple(a), tuple(b)
        if ka not in seen_source and kb not in seen_target:
            kept.append(i)
            seen_source.add(ka)
            seen_target.add(kb)
    if len(kept) < 3:
        raise ValueError('degenerate correspondence geometry: too few distinct locations')
    for cloud in (source[kept], target[kept]):
        singular = np.linalg.svd((cloud - cloud.mean(axis=0)) / normalizer, compute_uv=False)
        if singular[1] <= max(singular[0] * 1e-10, 1e-12):
            raise ValueError('degenerate correspondence geometry')
    try:
        from dpvo import _teaser
    except ImportError as exc:
        raise ImportError('Optional TEASER++ backend is missing; run `pixi run build-teaser`') from exc
    start = perf_counter()
    result = _teaser.solve(source[kept] / normalizer, target[kept] / normalizer,
                           noise_bound / normalizer, int(workers), float(clique_time_limit))
    result['translation'] = np.asarray(result['translation']) * normalizer
    for key in ('clique_indices', 'translation_inlier_indices'):
        result[key] = [kept[i] for i in result[key]]
    result.update(normalization_range=normalizer, noise_bound=float(noise_bound),
                  input_count=len(source), solver_input_indices=kept,
                  coincident_locations_omitted=len(source) - len(kept),
                  solve_and_conversion_seconds=perf_counter() - start)
    return result


def verify_teaser_geometry(query, candidate, matches, *, noise_fraction=.03,
                           relative_threshold=.03, workers=4, clique_time_limit=2.):
    """Use unique-target appearance matches; retain the existing depth check.

    noise_fraction times the median target range defines ONE absolute solver
    noise bound. It is not equivalent to the final heteroscedastic 3% gate.
    Returned inliers indicate depth consistency only, never verified loop edges.
    """
    for value in (noise_fraction, relative_threshold):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('noise and residual fractions must be positive and finite')
    matches = np.asarray(matches)
    if matches.ndim != 2 or matches.shape[1:] != (2,) or matches.dtype.kind not in 'iu':
        raise ValueError('matches must be an integer (N, 2) array')
    if len(matches) and (matches.min() < 0 or matches[:, 0].max() >= len(query.points)
                        or matches[:, 1].max() >= len(candidate.points)):
        raise ValueError('match index out of bounds')
    inliers = np.zeros(len(matches), bool)
    result = dict(estimator='teaser++', depth_matches=0, geometric_inliers=0,
                  geometric_inlier_ratio=0., scale=None, rotation=None, translation=None,
                  median_relative_residual=None, verification_status='no_matches',
                  noise_fraction=float(noise_fraction), relative_threshold=float(relative_threshold))
    if not len(matches):
        return result, inliers
    source = np.asarray(query.points[matches[:, 0]], np.float64)
    target = np.asarray(candidate.points[matches[:, 1]], np.float64)
    valid = (np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
             & (np.linalg.norm(source, axis=1) > 0) & (np.linalg.norm(target, axis=1) > 0))
    source, target = source[valid], target[valid]
    result['depth_matches'] = len(source)
    if len(source) < 6:
        result['verification_status'] = 'insufficient_depth_matches'
        return result, inliers
    bound = noise_fraction * float(np.median(np.linalg.norm(target, axis=1)))
    try:
        model = estimate_sim3(source, target, noise_bound=bound, workers=workers,
                              clique_time_limit=clique_time_limit)
    except ValueError as exc:
        if 'degenerate' not in str(exc):
            raise
        result['verification_status'] = 'degenerate_geometry'
        return result, inliers
    # All solver metadata are retained even when its model fails the common gate.
    result['solver'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}
    s, R, t = model['scale'], model['rotation'], model['translation']
    if (not model['valid'] or not np.isfinite(s) or not np.isfinite(R).all()
            or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-5)):
        result['verification_status'] = 'invalid_solver_model'
        return result, inliers
    result.update(scale=float(s), rotation=R.tolist(), translation=t.tolist())
    if not .25 <= s <= 4:
        result['verification_status'] = 'scale_out_of_range'
        return result, inliers
    clique = model['clique_indices']
    if len(clique) < 3 or any(np.linalg.matrix_rank(cloud[clique]-cloud[clique].mean(axis=0),tol=1e-10)<2
                             for cloud in (source,target)):
        result['verification_status'] = 'degenerate_solver_clique'
        return result, inliers
    residual = np.linalg.norm(s * (source @ R.T) + t - target, axis=1)
    relative = residual / np.linalg.norm(target, axis=1)
    support = relative < relative_threshold
    result['model_support_before_minimum'] = int(support.sum())
    if support.sum() < 6:
        result['verification_status'] = 'insufficient_consensus'
        return result, inliers
    inliers[np.flatnonzero(valid)] = support
    result.update(verification_status='checked', geometric_inliers=int(support.sum()),
                  geometric_inlier_ratio=float(support.mean()),
                  median_relative_residual=float(np.median(relative[support])))
    return result, inliers

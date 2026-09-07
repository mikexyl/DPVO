"""Inspect saved sphere descriptors before/after appearance and depth filtering.

Reuses feature archives and the frozen retrieval ranking. No extraction,
vocabulary training, or model inference is performed.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dpvo.sphere_bow import (DESCRIPTOR_VERSIONS, SphereFeatures, match_features,
                             validate_descriptor_tag, verify_depth_geometry)
from sphere_place_recognition import pair_image, read_sphere, save_rgb


def load_features(path, family, version):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].item() if archive[key].ndim == 0 else archive[key].copy()
                  for key in archive.files}
    validate_descriptor_tag(values, family, version)
    values.setdefault('face_ids', None)
    values.setdefault('face_xy', None)
    feature = SphereFeatures(**values)
    n = len(feature.descriptors)
    if feature.descriptors.dtype != np.uint8 or feature.descriptors.shape != (n, 32):
        raise ValueError(f'invalid descriptor archive: {path}')
    if feature.uv.shape != (n, 2) or not np.isfinite(feature.uv).all():
        raise ValueError(f'invalid panorama coordinates: {path}')
    return feature


def nearest_matches(query, candidate):
    """One-way Hamming nearest neighbor, before every acceptance filter."""
    validate_descriptor_tag(vars(candidate), query.descriptor_family, query.descriptor_version)
    if not len(query.descriptors) or not len(candidate.descriptors):
        return np.empty((0, 2), np.int32), np.empty(0, np.uint16)
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING).match(query.descriptors, candidate.descriptors),
                     key=lambda m: m.queryIdx)
    return (np.asarray([(m.queryIdx, m.trainIdx) for m in matches], np.int32).reshape(-1, 2),
            np.asarray([m.distance for m in matches], np.uint16))


def match_segments(query, candidate, matches, width, height):
    a, b = query.uv[matches[:, 0]].copy(), candidate.uv[matches[:, 1]].copy()
    a[:, 0] %= width
    b[:, 0] %= width
    a[:, 1] += 36
    b[:, 1] += height + 72
    return np.stack((a, b), axis=1).astype(np.float32)


def depth_display_masks(query, candidate, matches, geometry, inliers, evaluated=None):
    """Separate tested outliers from matches lacking a fitted depth model."""
    checked = np.zeros(len(matches), bool)
    if geometry['verification_status'] == 'checked':
        checked = (np.isfinite(query.points[matches[:, 0]]).all(axis=1)
                   & np.isfinite(candidate.points[matches[:, 1]]).all(axis=1))
        if evaluated is not None:
            checked &= evaluated
    return checked & ~inliers, ~checked


def export(retrieval, output, query_anchor=None):
    import rerun as rr
    import rerun.blueprint as rrb

    report = json.loads((retrieval / 'report.json').read_text())
    root = Path(report.get('visualization_input', report['input'])).resolve()
    snapshots = json.loads((root / 'manifest.json').read_text())['snapshots']
    family = report.get('descriptor_family', 'cube-orb')
    version = report.get('descriptor_version', DESCRIPTOR_VERSIONS[family])
    queries = [q for q in report['queries'] if query_anchor is None or q['query_anchor'] == query_anchor]
    if not queries:
        raise ValueError('no eligible query found')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'matches').mkdir()
    rr.init(f'{family}: raw and depth-checked sphere matches', strict=True)
    rr.save(str(output / 'sphere_matches.rrd'))

    def view(name, paths):
        return rrb.Spatial2DView(name=name, origin='pair',
                                 contents=['$origin/background', *[f'$origin/{p}/**' for p in paths]])

    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(
            view('All appearance matches — before depth', ['appearance']),
            rrb.Tabs(view('Depth: green inliers / magenta outliers / gray unverified', ['inliers', 'outliers', 'unverified']),
                     view('Unfiltered nearest neighbors — before ratio/mutual filters', ['nearest']),
                     active_tab=0), column_shares=[1, 1]),
        rrb.TextDocumentView(name='Match counts and filter definitions', origin='info'),
        row_shares=[.88, .12]), collapse_panels=True), make_active=True, make_default=True)

    features, images = {}, {}

    def get(index):
        if index not in features:
            anchor = snapshots[index]['anchor_timestamp']
            features[index] = load_features(retrieval / 'features' / f'{anchor:06d}.npz', family, version)
            images[index], _ = read_sphere(root, snapshots[index], report['settings'].get('downsample', 1))
        return features[index], images[index]

    records = []
    for query in queries:
        i = query['query_index']
        selected = query['candidates'][0]
        j = selected['index']
        if snapshots[i]['anchor_timestamp'] != query['query_anchor'] or snapshots[j]['anchor_timestamp'] != selected['anchor']:
            raise ValueError('retrieval and manifest anchors disagree')
        if j >= i or not set(snapshots[i]['source_timestamps']).isdisjoint(snapshots[j]['source_timestamps']):
            raise ValueError('invalid candidate provenance')
        q, q_image = get(i)
        c, c_image = get(j)
        raw, distances = nearest_matches(q, c)
        appearance = match_features(q, c, ratio=report['settings'].get('match_ratio', .75), gate=report['settings'].get('match_gate', 'strict'),
                                    patch_radius_degrees=report['settings'].get('patch_radius_degrees', 1.))
        geometry, inliers = verify_depth_geometry(q, c, appearance, seed=report['settings']['seed'])
        rejected, unverified = depth_display_masks(q, c, appearance, geometry, inliers)
        if len(appearance) != selected['mutual_matches'] or geometry['geometric_inliers'] != selected['geometric_inliers']:
            raise RuntimeError('recomputed matches differ from the saved report')
        height, width = q_image.shape[:2]
        if c_image.shape != q_image.shape:
            raise ValueError('query and candidate image dimensions differ')
        title = f"{family} | query {query['query_anchor']} -> candidate {selected['anchor']}"
        subtitle = f'{len(raw)} nearest neighbors | {len(appearance)} appearance matches | {inliers.sum()} depth inliers'
        empty = np.empty((0, 2), np.int32)
        background = pair_image(q_image, c_image, q, c, empty, np.zeros(0, bool), title, subtitle)
        rr.set_time('sphere_anchor', sequence=query['query_anchor'])
        rr.log('pair/background', rr.Image(background))
        raw_segments = match_segments(q, c, raw, width, height)
        segments = match_segments(q, c, appearance, width, height)
        for name, lines, color, radius in (
            ('appearance', segments, [255, 175, 50, 220], .8),
            ('inliers', segments[inliers], [80, 240, 100, 230], .9),
            ('outliers', segments[rejected], [255, 65, 190, 255], 1.4),
            ('unverified', segments[unverified], [180, 180, 180, 255], 1.2),
            ('nearest', raw_segments, [90, 200, 255, 30], .45)):
            rr.log(f'pair/{name}/lines', rr.LineStrips2D(lines, colors=color, radii=rr.Radius.ui_points(radius),
                                                       draw_order=10 if name == 'outliers' else 5))
            if name != 'nearest':
                rr.log(f'pair/{name}/endpoints', rr.Points2D(lines.reshape(-1, 2), colors=color,
                                                          radii=rr.Radius.ui_points(2), draw_order=11))
        info = [title, subtitle,
                f'Left: every appearance match in orange; {int(rejected.sum())} rejected by depth, {int(unverified.sum())} not verified.',
                f'Appearance gate: {report["settings"].get("match_gate", "strict")}; Hamming distance <=64, ratio threshold {report["settings"].get("match_ratio", .75)} when enabled.',
                f'Depth status: {geometry["verification_status"]}. Sim(3) fits unrestricted 3D rotation, translation and scale; at least 6 valid-depth matches required.',
                'Right: depth inliers green, tested outliers magenta, unverified gray. Switch its tab for every unfiltered nearest neighbor.',
                'Unfiltered nearest neighbors use one-way Hamming best matches without ratio, distance or mutual tests; duplicates are allowed.',
                'All lines are logged, with no display cap. Hide individual line/endpoint entities in Rerun as needed.',
                'Scrub sphere_anchor to inspect other top-ranked candidates. Geometry is diagnostic; no loop constraints are inserted.']
        rr.log('info', rr.TextDocument('\n'.join(info)))
        stem = f"{query['query_anchor']:06d}_{selected['anchor']:06d}"
        np.savez_compressed(output / 'matches' / f'{stem}.npz', nearest_matches=raw, nearest_hamming=distances,
                            appearance_matches=appearance, depth_inliers=inliers, depth_outliers=rejected,
                            depth_unverified=unverified, verification_status=geometry['verification_status'],
                            descriptor_family=family, descriptor_version=version)
        appearance_png = pair_image(q_image, c_image, q, c, appearance, np.zeros(len(appearance), bool),
                                    title, f'ALL {len(appearance)} appearance matches, before depth filtering')
        depth_png = pair_image(q_image, c_image, q, c, appearance, inliers,
                              title, f'{inliers.sum()} inliers, {int(rejected.sum())} rejected, {int(unverified.sum())} unverified (gray)',
                              unverified=unverified)
        save_rgb(output / 'matches' / f'{stem}_appearance.png', appearance_png)
        save_rgb(output / 'matches' / f'{stem}_depth.png', depth_png)
        records.append(dict(query_anchor=query['query_anchor'], candidate_anchor=selected['anchor'],
                            nearest_neighbors=len(raw), appearance_matches=len(appearance),
                            depth_inliers=int(inliers.sum()), rejected_by_depth=int(rejected.sum()),
                            not_verified=int(unverified.sum()), verification_status=geometry['verification_status']))
    rr.get_global_data_recording().flush()
    rr.disconnect()
    result = dict(input=str(retrieval.resolve()), descriptor_family=family, descriptor_version=version,
                  ranking='saved top-one candidate, unchanged', displayed_match_cap=None,
                  views=['all appearance matches before depth', 'depth inliers, outliers and unverified matches',
                         'all one-way nearest neighbors before appearance filtering'], queries=records)
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Saved {len(records)} candidate pairs: {output / 'sphere_matches.rrd'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Fresh output directory')
    parser.add_argument('--query-anchor', type=int, help='Optional single query; otherwise export every eligible query')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    cv2.setNumThreads(args.threads)
    export(args.retrieval.resolve(), args.output.resolve(), args.query_anchor)


if __name__ == '__main__':
    main()

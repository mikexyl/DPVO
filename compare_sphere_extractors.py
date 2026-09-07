"""Offline cube ORB/SPHORB comparison on saved spheres; never runs learned models."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from dpvo.sphere_bow import match_features, panorama_uv
from plot_sphere_loop_graph import build_graph, plot_graph, save_rerun
from sphere_place_recognition import (display_rgb, feature_image, make_extractor, parser as retrieval_parser,
                                      read_sphere, run, save_rgb)


def labeled_image(picture, label):
    picture = cv2.copyMakeBorder(picture, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    cv2.putText(picture, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    return picture


def pixel_rays(width, height):
    y, x = np.mgrid[:height, :width]
    lon = ((x + .5) / width - .5) * (2 * np.pi)
    lat = ((y + .5) / height - .5) * np.pi
    return np.stack((np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)), axis=-1)


def analytic_texture(rays):
    """Continuous 3D sinusoidal field on S2, with no panorama seam or special poles."""
    rng = np.random.default_rng(91)
    value = np.zeros(rays.shape[:-1], np.float64)
    for _ in range(24):
        direction = rng.normal(size=3)
        direction *= rng.uniform(50, 220) / np.linalg.norm(direction)
        value += 18 * np.sin(np.einsum('...i,i->...', rays, direction) + rng.uniform(-np.pi, np.pi))
    gray = np.clip(value + 128, 0, 255).astype(np.uint8)
    return np.dstack((gray, gray, gray, np.full_like(gray, 255)))


def rotated_panorama(bgra, rotation, rays):
    uv = panorama_uv(rays @ rotation, bgra.shape[1], bgra.shape[0]).astype(np.float32)
    uv[..., 1] = np.clip(uv[..., 1], 0, bgra.shape[0] - 1)
    clean = bgra[..., :3].copy()
    clean[bgra[..., 3] != 255] = 0
    color = cv2.remap(clean, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    invalid = cv2.remap((bgra[..., 3] != 255).astype(np.float32), uv[..., 0], uv[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return np.dstack((color, (invalid == 0).astype(np.uint8) * 255))


def rotation_metrics(a, b, rotation, tolerance_degrees=1.):
    expected = a.bearings @ rotation.T
    dots = np.clip(expected @ b.bearings.T, -1, 1)
    threshold = np.cos(np.deg2rad(tolerance_degrees))
    # Greedy one-to-one geometric correspondence, best angular distance first.
    rows, cols = np.nonzero(dots >= threshold)
    order = np.argsort(-dots[rows, cols], kind='stable')
    used_a, used_b = set(), set()
    for k in order:
        i, j = int(rows[k]), int(cols[k])
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    repeated = np.zeros(len(a.descriptors), bool)
    repeated[list(used_a)] = True
    matches = match_features(a, b)
    errors = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', expected[matches[:, 0]],
                                                  b.bearings[matches[:, 1]]), -1, 1)))
    correct = errors <= tolerance_degrees
    result = dict(source_features=len(a.descriptors), target_features=len(b.descriptors),
                  repeated_features=len(used_a), repeatability=len(used_a) / max(1, min(len(a.descriptors), len(b.descriptors))),
                  mutual_matches=len(matches), correct_matches=int(correct.sum()),
                  matching_precision=float(correct.mean()) if len(matches) else None,
                  median_match_error_degrees=float(np.median(errors)) if len(errors) else None,
                  tolerance_degrees=tolerance_degrees)
    lon = np.arctan2(expected[:, 0], expected[:, 2])
    for name, region in (('seam', np.abs(lon) >= np.deg2rad(150)),
                          ('poles', np.abs(expected[:, 1]) >= np.sin(np.deg2rad(60)))):
        selected = region[matches[:, 0]]
        result[name] = dict(source_features=int(region.sum()), repeated_features=int((region & repeated).sum()),
                            mutual_matches=int(selected.sum()), correct_matches=int((selected & correct).sum()),
                            matching_precision=float(correct[selected].mean()) if selected.any() else None)
    return result


def rotation_evaluation(root, snapshots, settings, output):
    rotations = {'identity': np.eye(3),
                 'yaw_72': cv2.Rodrigues(np.array([0., np.deg2rad(72), 0.]))[0],
                 'seam_yaw_173': cv2.Rodrigues(np.array([0., np.deg2rad(173), 0.]))[0],
                 'pole_pitch_90': cv2.Rodrigues(np.array([np.pi / 2, 0., 0.]))[0],
                 'roll_45': cv2.Rodrigues(np.array([0., 0., np.pi / 4]))[0],
                 'oblique': cv2.Rodrigues(np.deg2rad(np.array([31., 57., 113.])))[0]}
    extractors = {name: make_extractor(args) for name, args in settings.items()}
    rays = pixel_rays(1280, 640)
    sources = [('analytic_full_sphere', analytic_texture(rays), True)]
    for i in (0, len(snapshots) // 2, len(snapshots) - 1):
        bgra, _ = read_sphere(root, snapshots[i])
        sources.append((f'loris_{snapshots[i]["anchor_timestamp"]}', bgra, False))
    records = []
    (output / 'rotations').mkdir()
    for source, bgra, analytic in sources:
        source_rays = pixel_rays(bgra.shape[1], bgra.shape[0])
        depth = np.ones(bgra.shape[:2], np.float32)
        base = {name: extractor(bgra, depth) for name, extractor in extractors.items()}
        for label, rotation in rotations.items():
            target = (bgra.copy() if label == 'identity' else analytic_texture(source_rays @ rotation)
                      if analytic else rotated_panorama(bgra, rotation, source_rays))
            features = {name: extractor(target, depth) for name, extractor in extractors.items()}
            for name in extractors:
                records.append(dict(source=source, rotation_name=label, rotation_matrix=rotation.tolist(),
                                    extractor=name, **rotation_metrics(base[name], features[name], rotation)))
            if analytic:
                side = np.concatenate([labeled_image(feature_image(target, features[name]), f'{name} / {label}')
                                       for name in extractors], axis=1)
                save_rgb(output / 'rotations' / f'{label}.png', side)
        print(f'Rotation evaluation completed: {source}', flush=True)
    report = dict(records=records, protocol='Known active rotations in RDF; 1 degree, one-to-one geometric repeatability; mutual ratio .75 / Hamming <=64. Analytic images are rendered directly on S2. Saved partial spheres are bilinearly rotated with strict support. No depth enters these metrics.',
                  region_definitions='Seam: transformed |longitude| >=150 degrees; poles: transformed |latitude| >=60 degrees.',
                  caveat='Repeatability uses min(source,target) feature count; partial support changes under rotation. Procedural texture and three saved spheres do not establish general rotation robustness.')
    (output / 'rotations.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def aggregate(report):
    candidates = [c for q in report['queries'] for c in q['candidates']]
    top = [q['candidates'][0] for q in report['queries']]
    return dict(**report['summary'],
                warmed_extraction_ms=report['benchmark']['extraction_ms'],
                warmed_native_ms=report['benchmark']['native_ms'],
                warmed_wrapper_ms=report['benchmark']['wrapper_ms'],
                setup=report['benchmark']['setup'],
                file_io_ms=dict(median=float(np.median([f['file_io_seconds'] for f in report['features']]) * 1000),
                                p95=float(np.percentile([f['file_io_seconds'] for f in report['features']], 95) * 1000)),
                mean_occupied_angular_bins=float(np.mean([f['angular_distribution']['occupied_bins'] for f in report['features']])),
                total_pole_features=sum(f['angular_distribution']['pole_features'] for f in report['features']),
                top1_mutual_matches=sum(c['mutual_matches'] for c in top),
                top1_depth_inliers=sum(c['geometric_inliers'] for c in top),
                top3_mutual_matches=sum(c['mutual_matches'] for c in candidates),
                top3_depth_inliers=sum(c['geometric_inliers'] for c in candidates),
                top1_candidates_with_depth_inliers=sum(c['geometric_inliers'] > 0 for c in top))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spheres', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--keyframes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = args.spheres.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    snapshots = manifest['snapshots']
    reports, settings, graphs = {}, {}, {}
    # Run sequentially, including all warmed timing passes, to avoid backend CPU contention.
    for name in ('cube-orb', 'sphorb'):
        settings[name] = retrieval_parser().parse_args(['--spheres', str(root), '--extractor', name,
                           '--output', str(output / name), '--words', '4096', '--train-spheres', '12',
                           '--benchmark-passes', '3', '--threads', '4'])
        reports[name] = run(settings[name])
        graph = build_graph(np.loadtxt(args.trajectory, ndmin=2), json.loads(args.keyframes.read_text()),
                            manifest, reports[name], top_k=1)
        graphs[name] = graph
        graph_output = output / name / 'graph'
        graph_output.mkdir()
        (graph_output / 'loop_graph.json').write_text(json.dumps(graph, indent=2) + '\n')
        plot_graph(graph, graph_output, f'Loris / {name} / candidate loops')
        save_rerun(graph, graph_output)
    with np.load(output / 'cube-orb/retrieval.npz') as a, np.load(output / 'sphorb/retrieval.npz') as b:
        if not np.array_equal(a['eligible'], b['eligible']):
            raise RuntimeError('Backend candidate eligibility differs; comparison needs review')
    (output / 'overlays').mkdir()
    for snapshot in snapshots:
        anchor = snapshot['anchor_timestamp']
        pictures = []
        for name in reports:
            picture = cv2.imread(str(output / name / 'features' / f'{anchor:06d}.png'))
            pictures.append(labeled_image(picture, f'{name} / anchor {anchor}'))
        cv2.imwrite(str(output / 'overlays' / f'{anchor:06d}.png'), np.concatenate(pictures, axis=1))
    for filename, relative in [('similarity_side_by_side.png', 'similarity.png'),
                                ('candidate_graphs_side_by_side.png', 'graph/loop_graph.png')]:
        pictures = [labeled_image(cv2.imread(str(output / name / relative)), name) for name in reports]
        cv2.imwrite(str(output / filename), np.concatenate(pictures, axis=0))
    rotations = rotation_evaluation(root, snapshots, settings, output)
    summary = {name: aggregate(report) for name, report in reports.items()}
    common_pairs = {}
    for name, report in reports.items():
        common_pairs[name] = {(q['query_anchor'], c['anchor']): c for q in report['queries'] for c in q['candidates']}
    shared = common_pairs['cube-orb'].keys() & common_pairs['sphorb'].keys()
    shared_stats = {name: dict(pairs=len(shared), mutual_matches=sum(common_pairs[name][p]['mutual_matches'] for p in shared),
                               depth_inliers=sum(common_pairs[name][p]['geometric_inliers'] for p in shared)) for name in reports}
    # Decode and validate every recording, including image entities and geometry.
    verification = []
    for recording in sorted(output.rglob('*.rrd')):
        checked = subprocess.run(['rerun', 'rrd', 'verify', str(recording)], capture_output=True, text=True)
        verification.append(dict(path=str(recording.relative_to(output)), returncode=checked.returncode,
                                 stdout=checked.stdout, stderr=checked.stderr))
        if checked.returncode:
            raise RuntimeError(f'Recording verification failed: {recording}: {checked.stderr}')
    comparison = dict(summaries=summary, shared_ranked_pairs=shared_stats,
                      recording_verification=verification, sphere_count=len(snapshots),
                      rotation_report='rotations.json',
                      caveats=['All edges remain unverified candidates. Depth inliers use noisy DA3 geometry.',
                               'Different candidate rankings are compared; shared_ranked_pairs isolates pairs appearing in both top-three lists.',
                               'Three warmed passes per backend, sequentially executed. File I/O is separate from extraction.',
                               'No place ground truth: feature counts or depth matches alone do not measure retrieval accuracy.'])
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    lines = ['# Loris sphere feature comparison', '',
             '53 saved spheres; separate 4,096-word vocabularies from the first 12 spheres. All graph edges are candidates.', '',
             '| Metric | Cube ORB | SPHORB |', '|---|---:|---:|']
    for label, key in [('Mean features', 'mean_features'), ('Occupied equal-area bins / 72', 'mean_occupied_angular_bins'),
                       ('Top-1 mutual matches (sum)', 'top1_mutual_matches'), ('Top-1 depth inliers (sum)', 'top1_depth_inliers'),
                       ('Top-3 mutual matches (sum)', 'top3_mutual_matches'), ('Top-3 depth inliers (sum)', 'top3_depth_inliers')]:
        lines.append(f'| {label} | {summary["cube-orb"][key]:.2f} | {summary["sphorb"][key]:.2f} |')
    for label, key in [('Warmed median extraction, ms', 'median'), ('Warmed p95 extraction, ms', 'p95')]:
        lines.append(f'| {label} | {summary["cube-orb"]["warmed_extraction_ms"][key]:.2f} | {summary["sphorb"]["warmed_extraction_ms"][key]:.2f} |')
    lines += ['', 'The recordings passed `rerun rrd verify`. See `comparison.json` for timing breakdowns, common-pair metrics and verification output; `rotations.json` for known-rotation results.', '',
              'Appearance/depth counts are diagnostics, without place ground truth. No loop closures or model reruns were performed.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()

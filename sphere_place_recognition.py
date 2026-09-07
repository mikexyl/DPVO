"""Retrieve saved camera-centered spheres using masked ORB and binary TF-IDF."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from dpvo.sphere_bow import (FACE_NAMES, candidate_mask, extract_orb, fit_idf,
                           fit_vocabulary, match_features, rank_candidates,
                           tfidf, verify_depth_geometry, word_counts, DESCRIPTOR_VERSIONS,
                           validate_descriptor_tag, CubeOrbExtractor)


FACE_COLORS = np.asarray([[255, 100, 80], [80, 220, 120], [80, 160, 255],
                          [240, 200, 80], [220, 100, 240], [80, 220, 220]], np.uint8)


def downsample_sphere(bgra, depth, factor=1):
    """Area-average observed color; retain only fully observed output blocks.

    Radial depth uses a nearest pixel to each block center, never an average
    across foreground/background surfaces. Coordinates are in the new image.
    """
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError('downsample factor must be a positive integer')
    if bgra.ndim != 3 or bgra.shape[2] != 4 or bgra.dtype != np.uint8 or depth.shape != bgra.shape[:2]:
        raise ValueError('expected uint8 BGRA and matching radial depth')
    if factor == 1:
        return bgra, depth
    height, width = bgra.shape[:2]
    if height % factor or width % factor or min(height // factor, width // factor) < 2:
        raise ValueError('downsample factor must divide image dimensions and leave at least 2x2')
    size = (width // factor, height // factor)
    observed = bgra[..., 3] == 255
    valid = observed.reshape(size[1], factor, size[0], factor).all(axis=(1, 3))
    color = bgra[..., :3].copy()
    color[~observed] = 0
    color = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
    color[~valid] = 0
    reduced = np.dstack((color, valid.astype(np.uint8) * 255))
    # Exact pixel-center nearest sampling, with ties selecting the lower index.
    offset = (factor - 1) // 2
    radial = depth[offset::factor, offset::factor].astype(np.float32, copy=True)
    radial[~valid | ~np.isfinite(radial) | (radial <= 0)] = np.nan
    return reduced, radial


def read_sphere(root, snapshot, downsample=1):
    directory = (root / snapshot['directory']).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError(f"sphere directory escapes input root: {directory}")
    bgra = cv2.imread(str(directory / 'rgb.png'), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"missing or non-RGBA sphere: {directory / 'rgb.png'}")
    with np.load(directory / 'projection.npz', allow_pickle=False) as data:
        depth = data['radial_depth'].copy()
    return downsample_sphere(bgra, depth, downsample)


def display_rgb(bgra):
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    rgb[bgra[..., 3] == 0] = [28, 28, 28]
    return rgb


def feature_colors(features):
    if features.face_ids is not None:
        return FACE_COLORS[features.face_ids]
    palette = np.vstack((FACE_COLORS, [[245, 245, 245]])).astype(np.uint8)
    return palette[features.octaves % len(palette)]


def angular_distribution(features):
    rays = features.bearings
    longitude = np.arctan2(rays[:, 0], rays[:, 2])
    counts, _, _ = np.histogram2d(rays[:, 1], longitude, bins=(6, 12), range=((-1, 1), (-np.pi, np.pi)))
    return dict(equal_area_latitude_longitude_counts=counts.astype(int).tolist(),
                occupied_bins=int(np.count_nonzero(counts)), total_bins=72,
                pole_features=int(np.count_nonzero(np.abs(rays[:, 1]) > np.sin(np.deg2rad(60)))))


def feature_image(bgra, features):
    rgb = display_rgb(bgra)
    for uv, color in zip(features.uv, feature_colors(features)):
        x, y = np.rint(uv).astype(int)
        cv2.circle(rgb, (x % rgb.shape[1], y), 2, tuple(int(c) for c in color), 1, cv2.LINE_AA)
    return rgb


def pair_image(query, candidate, q_features, c_features, matches, inliers, title, subtitle, unverified=None):
    height, width = query.shape[:2]
    canvas = np.full((2 * (height + 36), width, 3), 28, np.uint8)
    canvas[36:36 + height] = display_rgb(query)
    canvas[height + 72:] = display_rgb(candidate)
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, height + 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
    order = np.argsort(~inliers, kind='stable')
    for k in order:
        q, c = matches[k]
        a = np.rint(q_features.uv[q]).astype(int) + [0, 36]
        b = np.rint(c_features.uv[c]).astype(int) + [0, height + 72]
        color = (80, 240, 100) if inliers[k] else (245, 165, 60)
        if unverified is not None and unverified[k]:
            color = (180, 180, 180)
        cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(a), 3, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(b), 3, color, 1, cv2.LINE_AA)
    return canvas


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed writing {path}")


def heatmap(scores, allowed, snapshots, train_count, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#303030')
    for ax, data, title in zip(axes, [scores, np.where(allowed, scores, np.nan)],
                               ['All pairs (includes shared source images)',
                                'Held-out queries / past-only / no shared source images']):
        plotted = ax.imshow(data, vmin=0, vmax=1, cmap=cmap, interpolation='nearest')
        ticks = np.unique(np.linspace(0, len(snapshots) - 1, min(8, len(snapshots))).astype(int))
        labels = [str(snapshots[t]['anchor_timestamp']) for t in ticks]
        ax.set_xticks(ticks, labels, rotation=45)
        ax.set_yticks(ticks, labels)
        ax.set(xlabel='Candidate sphere anchor (processed frame)', ylabel='Query sphere anchor', title=title)
        if train_count:
            ax.axhline(train_count - .5, color='orange', linewidth=1)
            ax.axvline(train_count - .5, color='orange', linewidth=1)
    fig.colorbar(plotted, ax=axes, label='Binary TF-IDF cosine similarity (not a probability)', shrink=.8)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def rerun_recording(output, snapshots, images, features, queries, scores, allowed, spawn):
    import rerun as rr
    import rerun.blueprint as rrb
    from dpvo.local_sphere import observed_sphere_mesh

    rr.init('DPVO sphere place recognition', strict=True)
    rr.save(str(output / 'sphere_retrieval.rrd'))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Vertical(rrb.Spatial3DView(name='Features on query sphere', origin='sphere'),
                     rrb.Spatial2DView(name='Query features (face or octave colors)', origin='query')),
        rrb.Spatial2DView(name='Top BoW candidate / correspondences', origin='match'),
        rrb.Vertical(rrb.Spatial2DView(name='Similarity matrices', origin='heatmap'),
                     rrb.TextDocumentView(name='Candidate ranking and caveats', origin='info')),
        column_shares=[1, 1, 1]), collapse_panels=True), make_active=True, make_default=True)
    rr.log('sphere', rr.ViewCoordinates.RDF, static=True)
    rr.log('heatmap', rr.Image(cv2.cvtColor(cv2.imread(str(output / 'similarity.png')), cv2.COLOR_BGR2RGB)), static=True)
    rr.log('scores/all', rr.Tensor(scores, dim_names=['query', 'candidate'], value_range=[0, 1]), static=True)
    rr.log('scores/eligible', rr.Tensor(np.where(allowed, scores, np.nan),
                                      dim_names=['query', 'candidate'], value_range=[0, 1]), static=True)
    for record in queries:
        i = record['query_index']
        rr.set_time('sphere_anchor', sequence=snapshots[i]['anchor_timestamp'])
        rr.log('query', rr.Image(display_rgb(images[i])))
        rr.log('query/orb', rr.Points2D(features[i].uv, colors=feature_colors(features[i]), radii=2))
        vertices, triangles, uv = observed_sphere_mesh(images[i][..., 3] > 0, 1)
        rr.log('sphere/mesh', rr.Mesh3D(vertex_positions=vertices, triangle_indices=triangles,
                                       vertex_texcoords=uv, albedo_texture=display_rgb(images[i])))
        rr.log('sphere/orb', rr.Points3D(features[i].bearings * 1.006,
                                        colors=feature_colors(features[i]), radii=.006))
        rr.log('match', rr.Image(cv2.cvtColor(cv2.imread(str(output / record['match_image'])), cv2.COLOR_BGR2RGB)))
        lines = [f"Query anchor {record['query_anchor']}: {len(features[i].descriptors)} features",
                 'Past spheres only; no shared source keyframes, including with vocabulary training.',
                 'Ranking: binary TF-IDF cosine; not a loop-closure probability.', '',
                 'Candidate | BoW score | appearance matches | depth Sim3 inliers']
        for c in record['candidates']:
            lines.append(f"{c['anchor']} | {c['score']:.3f} | {c['mutual_matches']} | {c['geometric_inliers']}")
        lines += ['', 'Green lines: depth-consistent correspondences. Orange: appearance matches only.',
                  'Geometry uses noisy DA3 radial depths, not ground-truth poses.',
                  'No loop closures are inserted into DPVO. Repeated doors/walls can alias.',
                  'Scrub sphere_anchor to inspect earlier queries.']
        rr.log('info', rr.TextDocument('\n'.join(lines)))
    rr.get_global_data_recording().flush()
    rr.disconnect()
    if spawn:
        # Open the complete saved recording rather than race a short-lived producer.
        import subprocess
        subprocess.Popen(['rerun', '--new', '--detach-process', str(output / 'sphere_retrieval.rrd')],
                         start_new_session=True)


def run(args, prepared=None, visualization_input=None):
    started = perf_counter()
    if args.threads < 1 or args.benchmark_passes < 0 or args.downsample < 1:
        raise ValueError('threads/downsample must be positive and benchmark passes nonnegative')
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(args.seed)
    root = args.spheres.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    snapshots = manifest['snapshots']
    if prepared is not None:
        snapshots = [s for s in snapshots if s['anchor_timestamp'] in prepared]
        if args.benchmark_passes or args.downsample != 1:
            raise ValueError('prepared feature runs require benchmark-passes=0 and downsample=1')
    if len(snapshots) < 2:
        raise ValueError('need at least two saved spheres')
    if not args.vocabulary and not 1 <= args.train_spheres < len(snapshots):
        raise ValueError('--train-spheres must leave at least one later query sphere')
    if args.face_size < 96 or args.features_per_face < 1 or args.top_k < 1 or args.words < 2:
        raise ValueError('face size >= 96, positive feature/top-k budgets, and >= 2 words required')
    output = args.output or root / ('orb_bow' if args.extractor == 'cube-orb' else 'sphorb_bow')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'features').mkdir()
    (output / 'matches').mkdir()
    images, depths, features, feature_stats = [], [], [], []
    extractor = make_extractor(args) if prepared is None else None
    setup_timing = getattr(extractor, 'setup_timing', {})
    shape = None
    for i, snapshot in enumerate(snapshots):
        t = perf_counter()
        bgra, depth = read_sphere(visualization_input or root, snapshot)
        io_seconds = perf_counter() - t
        source_shape = bgra.shape
        preprocessing_start = perf_counter()
        bgra, depth = downsample_sphere(bgra, depth, args.downsample)
        preprocessing_seconds = perf_counter() - preprocessing_start
        if not args.no_rerun and (bgra.shape[0] % 2 or bgra.shape[1] % 2):
            raise ValueError('Rerun sphere mesh requires even output width and height')
        extraction_start = perf_counter()
        if shape is not None and bgra.shape != shape:
            raise ValueError('all spheres must have the same resolution')
        shape = bgra.shape
        if prepared is None:
            feature = extractor(bgra, depth)
            extraction_seconds = perf_counter() - extraction_start
        else:
            feature = prepared[snapshot['anchor_timestamp']]['features']
            validate_descriptor_tag(vars(feature), args.extractor, DESCRIPTOR_VERSIONS[args.extractor])
            extraction_seconds = prepared[snapshot['anchor_timestamp']]['extraction_seconds']
        seconds = perf_counter() - t
        if prepared is not None:
            seconds = io_seconds + preprocessing_seconds + extraction_seconds
        anchor = snapshot['anchor_timestamp']
        feature.save(output / 'features' / f'{anchor:06d}.npz')
        save_rgb(output / 'features' / f'{anchor:06d}.png', feature_image(bgra, feature))
        images.append(bgra)
        depths.append(depth)
        features.append(feature)
        feature_stats.append(dict(anchor=anchor, features=len(feature.descriptors),
                                  source_size=[source_shape[1], source_shape[0]],
                                  input_size=[bgra.shape[1], bgra.shape[0]],
                                  preprocessing_seconds=preprocessing_seconds,
                                  per_face=(dict(zip(FACE_NAMES, np.bincount(feature.face_ids, minlength=6).tolist()))
                                            if feature.face_ids is not None else None),
                                  per_octave=np.bincount(feature.octaves, minlength=7).tolist(),
                                  angular_distribution=angular_distribution(feature),
                                  file_io_seconds=io_seconds, extraction_seconds=extraction_seconds,
                                  load_and_extract_seconds=seconds))
        print(f"Sphere {i + 1}/{len(snapshots)}, anchor {anchor}: {len(feature.descriptors)} {args.extractor} ({seconds * 1000:.1f} ms)", flush=True)
    warmed = []
    for pass_index in range(args.benchmark_passes):
        for i, (bgra, depth) in enumerate(zip(images, depths)):
            t = perf_counter()
            feature = extractor(bgra, depth)
            elapsed = perf_counter() - t
            if not np.array_equal(feature.descriptors, features[i].descriptors) or not np.array_equal(feature.uv, features[i].uv):
                raise RuntimeError('nondeterministic warmed extraction')
            warmed.append(dict(pass_index=pass_index, sphere_index=i, extraction_seconds=elapsed,
                               **{k: v for k, v in getattr(extractor, 'last_timing', {}).items() if k != 'extraction_seconds'}))
        print(f'Completed warmed extraction pass {pass_index + 1}/{args.benchmark_passes}', flush=True)
    benchmark = dict(setup=setup_timing, passes=args.benchmark_passes, samples=warmed,
                     note='File I/O and optional downsampling are measured separately on the initial read; warmed passes use memory-resident preprocessed inputs. Cube native time sums OpenCV API calls; its wrapper includes Python/NumPy. SPHORB native time covers the C++ core; its wrapper includes preparation/export/depth projection.')
    for key in ('extraction_seconds', 'native_seconds', 'wrapper_seconds'):
        values = [r[key] * 1000 for r in warmed if key in r]
        benchmark[key.replace('_seconds', '_ms')] = (dict(median=float(np.median(values)), p95=float(np.percentile(values, 95))) if values else None)
    descriptors = [f.descriptors for f in features]
    t = perf_counter()
    if args.vocabulary:
        with np.load(args.vocabulary, allow_pickle=False) as vocab:
            validate_descriptor_tag(vocab, args.extractor, DESCRIPTOR_VERSIONS[args.extractor])
            centers, idf = vocab['centers'].copy(), vocab['idf'].copy()
            if centers.dtype != np.uint8 or centers.ndim != 2 or centers.shape[1] != 32:
                raise ValueError('vocabulary must contain uint8 [words,32] ORB centers')
            if idf.shape != (len(centers),) or not np.isfinite(idf).all() or (idf <= 0).any():
                raise ValueError('invalid vocabulary IDF')
            # Preserve/purge known same-sequence training provenance when reusing.
            training_root = str(vocab['training_root'].item())
            vocabulary_train_count = int(vocab['train_count'])
            same_sequence = training_root == str(root)
            train_count = vocabulary_train_count if same_sequence else 0
            if train_count < 0 or train_count > len(snapshots):
                raise ValueError('vocabulary training count does not match this sequence')
        training = dict(vocabulary=str(args.vocabulary.resolve()), same_sequence=same_sequence,
                        note='External vocabulary independence is the caller\'s responsibility.')
    else:
        train_count = args.train_spheres
        training_root, vocabulary_train_count = str(root), train_count
        centers = fit_vocabulary(descriptors[:train_count], args.words, seed=args.seed)
        idf = fit_idf(word_counts(descriptors[:train_count], centers))
        training = dict(vocabulary='binary k-medians trained on initial spheres only',
                        training_anchors=[s['anchor_timestamp'] for s in snapshots[:train_count]])
    vocabulary_seconds = perf_counter() - t
    np.savez_compressed(output / 'vocabulary.npz', centers=centers, idf=idf,
                        training_root=training_root, train_count=vocabulary_train_count,
                        descriptor_family=args.extractor, descriptor_version=DESCRIPTOR_VERSIONS[args.extractor])
    t = perf_counter()
    counts = word_counts(descriptors, centers)
    vectors = tfidf(counts, idf)
    scores = np.clip(vectors @ vectors.T, 0, 1)
    allowed = candidate_mask(snapshots, train_count, args.min_anchor_gap)
    nonempty = np.asarray([bool(len(d)) for d in descriptors])
    allowed &= nonempty[:, None] & nonempty[None, :]
    rankings = rank_candidates(scores, allowed, args.top_k)
    retrieval_seconds = perf_counter() - t
    np.savez_compressed(output / 'retrieval.npz', counts=counts, vectors=vectors, scores=scores,
                        eligible=allowed, anchors=[s['anchor_timestamp'] for s in snapshots])
    heatmap(scores, allowed, snapshots, train_count, output / 'similarity.png')
    queries = []
    for i, ranking in enumerate(rankings):
        if not ranking:
            continue
        candidates = []
        for rank, j in enumerate(ranking):
            matches = match_features(features[i], features[j], ratio=args.match_ratio, gate=args.match_gate,
                                     patch_radius_degrees=args.patch_radius_degrees)
            geometry, inliers = verify_depth_geometry(features[i], features[j], matches, seed=args.seed)
            candidate = dict(index=int(j), anchor=snapshots[j]['anchor_timestamp'], score=float(scores[i, j]),
                             appearance_matches=len(matches), mutual_matches=len(matches), **geometry)
            candidates.append(candidate)
            if rank == 0:
                match_image = f'matches/{snapshots[i]["anchor_timestamp"]:06d}_{snapshots[j]["anchor_timestamp"]:06d}.png'
                rgb = pair_image(images[i], images[j], features[i], features[j], matches, inliers,
                                 f'Query {snapshots[i]["anchor_timestamp"]} -> candidate {candidate["anchor"]} | BoW {candidate["score"]:.3f}',
                                 f'{len(matches)} appearance matches | {geometry["geometric_inliers"]} depth inliers | no shared source frames')
                save_rgb(output / match_image, rgb)
        queries.append(dict(query_index=i, query_anchor=snapshots[i]['anchor_timestamp'],
                            match_image=match_image, candidates=candidates))
        best = candidates[0]
        print(f"Query {snapshots[i]['anchor_timestamp']} -> {best['anchor']}: BoW {best['score']:.3f}, "
              f"{best['mutual_matches']} matches, {best['geometric_inliers']} depth inliers", flush=True)
    summary = dict(spheres=len(snapshots), words=len(centers), eligible_queries=len(queries),
                   eligible_pairs=int(allowed.sum()), mean_features=float(np.mean([len(d) for d in descriptors])),
                   mean_load_extract_ms=float(np.mean([s['load_and_extract_seconds'] for s in feature_stats]) * 1000),
                   vocabulary_seconds=vocabulary_seconds, quantize_rank_seconds=retrieval_seconds,
                   total_seconds_before_rerun=perf_counter() - started)
    report = dict(method=f'{args.extractor} -> Hamming binary k-medians -> log-TF / smoothed-IDF / L2 cosine',
                  matching=dict(gate=args.match_gate, ratio=args.match_ratio, max_distance=64,
                                patch_radius_degrees=args.patch_radius_degrees,
                                legacy_field='mutual_matches is retained as an alias of appearance_matches'),
                  descriptor_family=args.extractor, descriptor_version=DESCRIPTOR_VERSIONS[args.extractor],
                  preprocessing=dict(downsample=args.downsample, source_size=feature_stats[0]['source_size'],
                                     input_size=feature_stats[0]['input_size'],
                                     color=('INTER_AREA; unknown RGB zeroed; require all block pixels observed'
                                            if args.downsample > 1 else 'unchanged'),
                                     depth=('nearest block-center sample, ties toward lower index; invalid remains NaN'
                                            if args.downsample > 1 else 'unchanged'),
                                     coordinates='UV and feature sizes in preprocessed image pixels; RDF bearings unchanged'),
                  benchmark=benchmark,
                  input=str(root), settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  training=training, summary=summary, features=feature_stats, queries=queries,
                  caveats=['Train/query windows are purged if any source keyframe overlaps.',
                           'Candidates are earlier spheres with disjoint source-keyframe sets.',
                           'Cosine similarity is not a loop-closure probability; repeated structures can alias.',
                           'Sim3 depth checks are diagnostic only; no DPVO pose/graph changes.',
                           'This single sequence is not a ground-truth place-recognition benchmark.'])
    if visualization_input is not None:
        report['visualization_input'] = str(visualization_input.resolve())
        report['preprocessing'] = dict(input='prepared features; preview pixels do not enter extraction',
                                      coordinates='UV is a display coordinate; descriptors use direct source-image geodesic grids')
        report['benchmark']['note'] = 'Prepared extraction: timings and source I/O are in the virtual-input report; file I/O above reads visualization previews only.'
        for stats in report['features']:
            stats['virtual_timing'] = prepared[stats['anchor']]['timing']
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.no_rerun:
        rerun_recording(output, snapshots, images, features, queries, scores, allowed, args.spawn)
    print(json.dumps(summary, indent=2))
    print(f'Results: {output}')
    return report


def make_extractor(args):
    if args.extractor == 'sphorb':
        from dpvo.sphorb import SphorbExtractor
        return SphorbExtractor(args.features, args.levels, args.threshold, args.threads, args.sphorb_tables)
    return CubeOrbExtractor(args.face_size, args.features_per_face)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extractor', choices=('cube-orb', 'sphorb'), default='cube-orb')
    p.add_argument('--features', type=int, default=3000, help='SPHORB total feature budget')
    p.add_argument('--levels', type=int, default=7, help='SPHORB pyramid levels (1..7)')
    p.add_argument('--threshold', type=int, default=20, help='SPHORB FAST threshold')
    from dpvo.sphere_match_gates import GATES
    p.add_argument('--match-gate', choices=GATES, default='strict', help='Appearance matching gate; strict preserves the baseline')
    p.add_argument('--match-ratio', type=float, default=.75, help='Appearance ratio threshold when enabled by the gate')
    p.add_argument('--patch-radius-degrees', type=float, default=1., help='Angular neighborhood for optional patch mutual gates')
    p.add_argument('--sphorb-tables', type=Path, help='Explicit lookup-table directory')
    p.add_argument('--benchmark-passes', type=int, default=0, help='Additional warmed extraction passes')
    p.add_argument('--downsample', type=int, default=1, help='Divide input width/height by this integer before extraction (2 = half size)')
    p.add_argument('--spheres', required=True, type=Path, help='LocalSphereBuilder output directory with manifest.json')
    p.add_argument('--output', type=Path, help='New output directory (default: SPHERES/orb_bow); never overwrites')
    p.add_argument('--train-spheres', type=int, default=12, help='Initial spheres for vocabulary/IDF; overlapping query windows are purged')
    p.add_argument('--vocabulary', type=Path, help='Reuse a vocabulary.npz from this script; no fitting on query data')
    p.add_argument('--words', type=int, default=4096)
    p.add_argument('--face-size', type=int, default=384)
    p.add_argument('--features-per-face', type=int, default=500)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--min-anchor-gap', type=int, default=0, help='Additional separation in processed-frame indices, not seconds')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--no-rerun', action='store_true', help='Skip saving the Rerun recording')
    p.add_argument('--spawn', action='store_true', help='Open the resulting Rerun recording')
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.threads < 1 or arguments.min_anchor_gap < 0 or (arguments.spawn and arguments.no_rerun):
        raise SystemExit('threads must be positive, anchor gap nonnegative, and --spawn requires Rerun')
    run(arguments)

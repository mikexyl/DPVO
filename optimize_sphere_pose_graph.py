"""Optimize saved keyframe Sim3 poses with frozen sphere-derived loop candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dpvo.sphere_pose_graph import (Graph, as_sim3, edge_statistics, from_tum,
                                    measurement, optimize_graph)


def read_json(path):
    return json.loads(Path(path).read_text())


def build_graph(args):
    trajectory = np.loadtxt(args.trajectory, ndmin=2)
    from_tum(trajectory)  # Validate ordering/quaternions before indexing.
    frames = read_json(args.keyframes)
    timestamps = np.asarray(frames.get('source_timestamps', [f['timestamp'] for f in frames.get('frames', [])]))
    if (timestamps.ndim != 1 or len(timestamps) < 2 or timestamps.dtype.kind not in 'iu'
            or (np.diff(timestamps) <= 0).any()):
        raise ValueError('keyframes must have strictly ordered integer timestamps')
    rows = {row[0]: row for row in trajectory}
    if any(t not in rows for t in timestamps):
        raise ValueError('keyframe missing from saved trajectory')
    initial = from_tum(np.asarray([rows[t] for t in timestamps]))
    lookup = {int(t): i for i, t in enumerate(timestamps)}
    snapshots = read_json(args.spheres)['snapshots']
    spheres = {s['anchor_timestamp']: s for s in snapshots}
    if len(spheres) != len(snapshots) or not set(spheres).issubset(lookup):
        raise ValueError('sphere anchors must be distinct saved keyframes')
    report = read_json(args.loops)
    if report['descriptor_family'] != 'sphorb-virtual':
        raise ValueError('this run expects frozen virtual-SPHORB sphere matches')
    retrieval = read_json(Path(report['retrieval']) / 'report.json')
    allowed = {(q['query_anchor'], q['candidates'][0]['anchor']) for q in retrieval['queries']}
    step = float(np.median(np.linalg.norm(np.diff(initial[:, :3], axis=0), axis=1)))
    if step <= 1e-10:
        raise ValueError('cannot infer translation noise from a stationary trajectory')
    settings = dict(method=args.method, robust_loss='block pseudo-Huber on loops; quadratic odometry',
                    robust_delta=args.robust_delta,
                    odometry_translation_sigma=args.odom_translation_fraction * step,
                    odometry_rotation_sigma_degrees=args.odom_rotation_degrees,
                    odometry_log_scale_sigma=args.odom_log_scale,
                    loop_translation_sigma='max(0.03 * median target range, 0.5 * median keyframe step)',
                    loop_rotation_sigma_degrees=args.loop_rotation_degrees,
                    loop_log_scale_sigma=args.loop_log_scale,
                    confidence='min(depth_inliers/30,1) * min(depth_inlier_ratio/0.5,1)',
                    median_keyframe_step=step,
                    weighting_note='Explicit heuristic weights, not calibrated covariances; no ground-truth tuning')
    odom_sigma = np.r_[np.full(3, settings['odometry_translation_sigma']),
                       np.full(3, np.deg2rad(args.odom_rotation_degrees)), args.odom_log_scale]
    source = list(range(1, len(initial)))
    target = list(range(len(initial)-1))
    states = as_sim3(initial)
    measurements = list((states[target].inv() * states[source]).data.numpy())
    sigmas = [odom_sigma.copy() for _ in source]
    flags, confidence = [False] * len(source), [1.] * len(source)
    edges = [dict(source=int(timestamps[i]), target=int(timestamps[j]), kind='sequential_saved_pose')
             for i, j in zip(source, target)]
    excluded, seen = [], set()
    from inspect_sphere_matches import load_features
    features = {}
    for row in report['records']:
        q, c = row['query'], row['candidate']
        if (q, c) in seen or q not in spheres or c not in spheres or c >= q or (q, c) not in allowed:
            raise ValueError('invalid, duplicate, or ineligible loop endpoint')
        seen.add((q, c))
        if not set(spheres[q]['source_timestamps']).isdisjoint(spheres[c]['source_timestamps']):
            raise ValueError('sphere pair shares source keyframes')
        method = row['methods'][args.method]
        geometry = method['geometry']
        if geometry['verification_status'] != 'checked':
            excluded.append(dict(source=q, target=c, reason=geometry['verification_status']))
            continue
        mask = np.asarray(method['inliers'], dtype=bool)
        z = measurement(geometry['rotation'], geometry['translation'], geometry['scale'])
        if mask.sum() != geometry['geometric_inliers'] or mask.sum() < 6:
            raise ValueError('saved depth support is inconsistent')
        archive = args.loops.parent / 'matches' / f'{q:06d}_{c:06d}.npz'
        with np.load(archive) as data:
            matches = data['unique_matches'].copy()
            np.testing.assert_array_equal(mask, data[args.method])
        if hashlib.sha256(matches.tobytes()).hexdigest() != row['correspondence_sha256']:
            raise ValueError('frozen correspondence checksum mismatch')
        for anchor in (q, c):
            if anchor not in features:
                features[anchor] = load_features(Path(report['retrieval'])/'features'/f'{anchor:06d}.npz',
                    report['descriptor_family'], report['descriptor_version'])
        a, b = features[q].points[matches[:, 0]], features[c].points[matches[:, 1]]
        # Check transform direction and final support against the actual saved 3D matches.
        predicted = z[7] * (a @ np.asarray(geometry['rotation']).T) + z[:3]
        ranges = np.linalg.norm(b, axis=1)
        valid = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
                 & (np.linalg.norm(a, axis=1) > 0) & (ranges > 0))
        expected_mask = valid & (np.linalg.norm(predicted-b, axis=1) < geometry['relative_threshold'] * ranges)
        np.testing.assert_array_equal(mask, expected_mask)
        sigma_t = max(.03 * float(np.median(ranges[valid])), .5 * step)
        sigma = np.r_[np.full(3, sigma_t), np.full(3, np.deg2rad(args.loop_rotation_degrees)), args.loop_log_scale]
        weight = min(mask.sum()/30., 1.) * min(geometry['geometric_inlier_ratio']/.5, 1.)
        source.append(lookup[q]); target.append(lookup[c]); measurements.append(z)
        sigmas.append(sigma); flags.append(True); confidence.append(weight)
        edges.append(dict(source=q, target=c, kind='sphere_loop_candidate', method=args.method,
                          appearance_matches=row['appearance_matches'], unique_target_matches=len(matches),
                          depth_inliers=int(mask.sum()), depth_inlier_ratio=geometry['geometric_inlier_ratio'],
                          measurement=z.tolist(), sigma=sigma.tolist(), confidence=weight,
                          local_point_support_rechecked=True))
    if not any(flags):
        raise ValueError('no depth-supported sphere candidates to optimize')
    graph = Graph(initial, np.asarray(source), np.asarray(target), np.asarray(measurements),
                  np.asarray(sigmas), np.asarray(flags), np.asarray(confidence))
    return graph, timestamps, spheres, edges, excluded, settings, report, retrieval


def statistics_summary(graph, stats):
    result = {}
    for name, selection in [('odometry', ~graph.loops), ('sphere_loops', graph.loops)]:
        result[name] = {key: dict(median=float(np.median(value[selection])),
                                 p95=float(np.percentile(value[selection], 95)),
                                 max=float(np.max(value[selection]))) for key, value in stats.items()}
    return result


def plots(output, graph, optimized, edges, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    positions = [graph.initial[:, :3], optimized[:, :3]]
    loop_edges = np.c_[graph.source[graph.loops], graph.target[graph.loops]]
    weights = np.asarray([e['after']['robust_weight'] for e in edges if e['kind']=='sphere_loop_candidate'])
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True, sharey=True)
    for ax, xyz, title in zip(axes[:2], positions, ['Original saved poses', 'Optimized Sim3 poses']):
        xy = xyz[:, [0, 2]]
        ax.plot(*xy.T, color='#245382', linewidth=1.5)
        ax.scatter(*xy.T, color='#245382', s=5, zorder=3)
        ax.add_collection(LineCollection(xy[loop_edges], colors=plt.cm.RdYlGn(weights), linewidths=1., alpha=.8))
        ax.scatter(*xy[0], marker='^', s=70, color='#17956f', zorder=4)
        ax.scatter(*xy[-1], marker='s', s=70, color='#9c3868', zorder=4)
        ax.set_title(title)
    for xyz, label, color in zip(positions, ['Original', 'Optimized'], ['#778391', '#147cb0']):
        axes[2].plot(xyz[:, 0], xyz[:, 2], color=color, label=label)
    axes[2].set_title('Overlay, same fixed first keyframe'); axes[2].legend()
    for ax in axes:
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)
        ax.set_xlabel('X (DPVO units)'); ax.set_ylabel('Z (DPVO units)')
    fig.suptitle(f'Office | {len(optimized)} saved keyframes, {len(loop_edges)} sphere loop candidates', y=.96)
    color_axes = fig.add_axes([.928, .18, .012, .64])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='RdYlGn'), cax=color_axes,
                 label='Final robust loop weight (not correctness probability)')
    fig.text(.5, .015, 'Camera-to-world Sim3; monocular units. Depth-supported loops remain candidates. No ground-truth accuracy claim.',
             ha='center', fontsize=10)
    fig.subplots_adjust(bottom=.15, top=.88, wspace=.3, right=.895, left=.055)
    fig.savefig(output/'pose_graph.png', dpi=180); fig.savefig(output/'pose_graph.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(optimized[:, 7]); axes[0].set_ylabel('Camera-to-world scale'); axes[0].set_xlabel('Saved keyframe index')
    axes[1].semilogy([v['seconds'] for v in summary['optimizer']['history']],
                     [v['cost'] for v in summary['optimizer']['history']])
    axes[1].set_xlabel('Optimization seconds'); axes[1].set_ylabel('Best robust objective')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'optimization.png', dpi=160); plt.close(fig)


def recording(output, graph, optimized, timestamps, spheres, edges, summary, report, retrieval, reference=None):
    import rerun as rr
    import rerun.blueprint as rrb
    from inspect_sphere_matches import load_features
    from sphere_place_recognition import pair_image, read_sphere
    rr.init('Office virtual sphere Sim3 pose graph' + (' with reference' if reference is not None else ''), strict=True)
    rr.save(str(output/'pose_graph.rrd'))
    tabs = [
        rrb.Vertical(rrb.Horizontal(rrb.Spatial3DView(name='Original', origin='original'),
                                    rrb.Spatial3DView(name='Optimized', origin='optimized')),
                     rrb.TextDocumentView(origin='summary'), row_shares=[.84, .16], name='Sim3 graph'),
        rrb.Spatial2DView(name='Before / after / overlay', origin='plot'),
        rrb.Vertical(rrb.Spatial2DView(name='Panned sphere matches: all appearance matches', origin='pair'),
                     rrb.TextDocumentView(origin='loop_info'), row_shares=[.8, .2], name='Sphere loop matches'),
        rrb.Spatial2DView(name='Scales and objective', origin='optimization')]
    if reference is not None:
        def reference_view(name, root, paths):
            return rrb.Spatial3DView(name=name, origin=root,
                                     contents=[f'$origin/{p}' for p in ['ground_truth', *paths]])
        tabs[:0] = [
            rrb.Vertical(rrb.Horizontal(
                reference_view('Original + reference','reference',['original']),
                reference_view('Optimized + reference','reference',['optimized']),
                reference_view('All trajectories','reference',['original','optimized'])),
                rrb.TextDocumentView(origin='reference_info'), row_shares=[.82,.18],
                name='Ground truth comparison'),
            rrb.Spatial2DView(name='Ground truth plot',origin='reference_plot'),
            rrb.Vertical(reference_view('One common alignment, fitted to original','shared_reference',['original','optimized']),
                         rrb.TextDocumentView(origin='shared_reference_info'),row_shares=[.85,.15],name='Common alignment')]
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs, active_tab=0),
        collapse_panels=True), make_active=True, make_default=True)
    if reference is not None:
        for root,prefix in [('reference',''),('shared_reference','shared_')]:
            rr.log(root,rr.ViewCoordinates.RUB,static=True)
            for name,points,color in [('ground_truth',reference['positions'],[40,170,75]),
                                      ('original',reference[prefix+'original'],[130,140,155]),
                                      ('optimized',reference[prefix+'optimized'],[25,125,220])]:
                rr.log(root+'/'+name,rr.LineStrips3D([points],colors=color,radii=rr.Radius.ui_points(2)),static=True)
        rr.log('reference_plot',rr.EncodedImage(path=output/'reference_comparison.png'),static=True)
        ref_report=reference['report'];metric=ref_report['independent_sim3']
        rr.log('reference_info',rr.TextDocument(
            f'Green: ScaleMaster refined reference. Gray: original DPVO. Blue: sphere-loop Sim3 PGO. {len(timestamps)} exact matched video frames.\n'
            f'Independent global Sim3 alignment: original ATE RMSE {metric["original"]["rmse_m"]:.3f} m; optimized {metric["optimized"]["rmse_m"]:.3f} m. '
            'Each uses one transform fitted across the same full set of keyframes; no local alignment.\n'
            'Reference is optimized_odometry.csv: ARKit poses refined with verified loops and SE3 PGO; not independent motion capture. '
            'Reference was used only for this display/evaluation, never to optimize the graph.'),static=True)
        rr.log('shared_reference_info',rr.TextDocument(
            'The original DPVO global Sim3 alignment is applied unchanged to both trajectories. This preserves their relative deformation.\n'
            f'Original RMSE {metric["original"]["rmse_m"]:.3f} m; optimized RMSE '
            f'{ref_report["shared_alignment"]["optimized_errors"]["rmse_m"]:.3f} m. '
            'The standard independently aligned ATE comparison is in the first tab.'),static=True)
    for name, poses in [('original', graph.initial), ('optimized', optimized)]:
        rr.log(name, rr.ViewCoordinates.RDF, static=True)
        rr.log(name+'/trajectory', rr.LineStrips3D([poses[:, :3]], colors=[60,130,190], radii=rr.Radius.ui_points(1.5)), static=True)
        rr.log(name+'/keyframes', rr.Points3D(poses[:, :3], colors=[60,130,190], radii=rr.Radius.ui_points(2)), static=True)
        anchor_indices = [i for i,t in enumerate(timestamps) if t in spheres]
        rr.log(name+'/sphere_anchors', rr.Points3D(poses[anchor_indices, :3], colors=[50,185,220],
               labels=[str(timestamps[i]) for i in anchor_indices], show_labels=False, radii=rr.Radius.ui_points(3)), static=True)
        for index in np.flatnonzero(graph.loops):
            e = edges[index]; w = e['after']['robust_weight']
            color = [int(230*(1-w)), int(60+160*w), 65]
            rr.log(f'{name}/loop_candidates/{e["source"]}_{e["target"]}',
                   rr.LineStrips3D([poses[[graph.source[index], graph.target[index]], :3]],
                                  colors=color, radii=rr.Radius.ui_points(1.2)), static=True)
    rr.log('plot', rr.EncodedImage(path=output/'pose_graph.png'), static=True)
    rr.log('optimization', rr.EncodedImage(path=output/'optimization.png'), static=True)
    text = (f'{len(timestamps)} saved depth keyframes / {sum(~graph.loops)} sequential edges / {sum(graph.loops)} sphere candidates. '
            f'Robust objective {summary["optimizer"]["initial_cost"]:.2f} -> {summary["optimizer"]["final_cost"]:.2f}.\n'
            'First Sim3 fixed; no camera projection or heading gate. Edge colors show final robust weight, not correctness. '
            'Original DPVO poses remain unchanged. Loop tab retains all appearance matches on panned spheres.')
    rr.log('summary', rr.TextDocument(text), static=True)
    root = Path(retrieval.get('visualization_input', retrieval['input']))
    visual = {s['anchor_timestamp']:s for s in read_json(root/'manifest.json')['snapshots']}
    lookup = {int(t):i for i,t in enumerate(timestamps)}
    records = {(r['query'], r['candidate']):r for r in report['records']}
    loop_rows = [e for e in edges if e['kind']=='sphere_loop_candidate']
    # Put the richest sphere match first; then show every inserted candidate.
    loop_rows.sort(key=lambda e: (-e['appearance_matches'], e['source']))
    for frame, e in enumerate(loop_rows):
        q, c = e['source'], e['target']; row = records[q, c]
        rr.set_time('loop', sequence=frame)
        qf, cf = [load_features(Path(report['retrieval'])/'features'/f'{a:06d}.npz',
                               report['descriptor_family'], report['descriptor_version']) for a in (q,c)]
        with np.load(Path(summary['inputs']['loops']['path']).parent/'matches'/f'{q:06d}_{c:06d}.npz') as data:
            raw, unique = data['appearance_matches'], data['unique_matches']
        unique_mask = np.asarray(row['methods'][summary['settings']['method']]['inliers'], bool)
        by_query = {int(i): bool(v) for i,v in zip(unique[:,0], unique_mask)}
        accepted = np.asarray([by_query.get(int(i), False) for i in raw[:,0]])
        unverified = np.asarray([int(i) not in by_query for i in raw[:,0]])
        qi, ci = [read_sphere(root, visual[a], retrieval['settings'].get('downsample',1))[0] for a in (q,c)]
        picture = pair_image(qi,ci,qf,cf,raw,accepted, f'Sphere loop candidate {q} -> {c}',
                             f'{len(raw)} appearance matches / {accepted.sum()} TEASER++ depth inliers / graph weight {e["after"]["robust_weight"]:.3f}',
                             unverified=unverified)
        rr.log('pair/matches', rr.Image(picture).compress(jpeg_quality=90))
        rr.log('loop_info', rr.TextDocument(json.dumps(e,indent=2)))
        for name, poses in [('original',graph.initial),('optimized',optimized)]:
            rr.log(name+'/selected_candidate', rr.LineStrips3D([poses[[lookup[q],lookup[c]], :3]],
                   colors=[250,165,30], radii=rr.Radius.ui_points(3)))
    rr.get_global_data_recording().flush(); rr.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--keyframes', type=Path, required=True, help='Saved keyframe audit or dense metadata JSON')
    parser.add_argument('--spheres', type=Path, required=True, help='Sphere manifest.json')
    parser.add_argument('--loops', type=Path, required=True, help='Frozen TEASER comparison report.json')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--method', default='teaser-0.03')
    parser.add_argument('--robust-delta', type=float, default=3.)
    parser.add_argument('--odom-translation-fraction', type=float, default=.1)
    parser.add_argument('--odom-rotation-degrees', type=float, default=1.)
    parser.add_argument('--odom-log-scale', type=float, default=.02)
    parser.add_argument('--loop-rotation-degrees', type=float, default=5.)
    parser.add_argument('--loop-log-scale', type=float, default=.1)
    parser.add_argument('--max-nfev', type=int, default=400)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    graph, timestamps, spheres, edges, excluded, settings, report, retrieval = build_graph(args)
    print(f'Graph: {len(timestamps)} nodes / {sum(~graph.loops)} odometry / {sum(graph.loops)} sphere candidates', flush=True)
    before = edge_statistics(graph, graph.initial, args.robust_delta)
    optimized, solver = optimize_graph(graph, delta=args.robust_delta, max_nfev=args.max_nfev,
                                       workers=args.workers, verbose=True)
    after = edge_statistics(graph, optimized, args.robust_delta)
    for i, edge in enumerate(edges):
        edge['before'] = {key:float(value[i]) for key,value in before.items()}
        edge['after'] = {key:float(value[i]) for key,value in after.items()}
    displacement = np.linalg.norm(optimized[:, :3] - graph.initial[:, :3], axis=1)
    summary = dict(settings=settings, optimizer=solver, nodes=len(timestamps), odometry_edges=int(sum(~graph.loops)),
                   sphere_loop_candidates=int(sum(graph.loops)), excluded_candidates=excluded,
                   before=statistics_summary(graph,before), after=statistics_summary(graph,after),
                   optimized_scale=dict(min=float(optimized[:,7].min()), max=float(optimized[:,7].max()),
                                        median=float(np.median(optimized[:,7]))),
                   position_change=dict(median=float(np.median(displacement)), max=float(displacement.max())),
                   low_weight_candidates=[dict(source=e['source'],target=e['target'],weight=e['after']['robust_weight'])
                                          for e in edges if e['kind']=='sphere_loop_candidate' and e['after']['robust_weight']<.1],
                   coordinates='S_i: camera RDF to world, p_world = scale * rotation * p_camera + translation',
                   residual='Log((S_target^-1 S_source) Z_target_source^-1), order translation, rotation, log-scale',
                   node_selection='Saved accepted dense keyframes; this Office run has no retained-keyframe audit for frames without saved depth',
                   backbone='Consecutive relative final saved poses, not the internal DPVO patch-factor graph',
                   scope='Offline experimental optimization; sphere edges are candidates, not ground-truth-confirmed closures',
                   inputs={key:dict(path=str(getattr(args,key).resolve()),
                                    sha256=hashlib.sha256(getattr(args,key).read_bytes()).hexdigest())
                           for key in ('trajectory','keyframes','spheres','loops')})
    np.savez_compressed(args.output/'graph.npz', timestamps=timestamps, initial=graph.initial, optimized=optimized,
                        source=graph.source,target=graph.target,measurements=graph.measurements,
                        sigmas=graph.sigmas,loops=graph.loops,confidence=graph.confidence)
    for name, values in [('initial',graph.initial),('optimized',optimized)]:
        np.savetxt(args.output/f'{name}_keyframes_sim3.txt', np.c_[timestamps,values], fmt='%.12g',
                   header='timestamp tx ty tz qx qy qz qw camera_to_world_scale')
        np.savetxt(args.output/f'{name}_keyframes.tum', np.c_[timestamps,values[:,:7]], fmt='%.12g',
                   header='timestamp tx ty tz qx qy qz qw; per-keyframe scale is in the companion sim3 file')
    (args.output/'edges.json').write_text(json.dumps(edges,indent=2,allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plots(args.output,graph,optimized,edges,summary)
    recording(args.output,graph,optimized,timestamps,spheres,edges,summary,report,retrieval)
    print(json.dumps({key:summary[key] for key in ('nodes','sphere_loop_candidates','optimized_scale','position_change')},indent=2),flush=True)
    print(f'Optimizer: {solver["message"]}; objective {solver["initial_cost"]:.6g} -> {solver["final_cost"]:.6g}',flush=True)


if __name__ == '__main__':
    main()

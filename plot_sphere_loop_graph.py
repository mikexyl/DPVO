"""Plot saved DPVO keyframes and sphere retrieval candidates, without optimizing poses."""

import argparse
import json
from pathlib import Path

import numpy as np


def build_graph(trajectory, keyframes, manifest, report, top_k=1):
    trajectory = np.asarray(trajectory)
    if trajectory.ndim != 2 or trajectory.shape[1] != 8 or not np.isfinite(trajectory).all():
        raise ValueError("expected a finite TUM trajectory with eight columns")
    if top_k < 1 or np.any(np.diff(trajectory[:, 0]) <= 0):
        raise ValueError("positive top-k and strictly increasing trajectory timestamps required")
    # Saved demo timestamps are stride-selected input indices, not raw video indices.
    poses = {float(row[0]): row[1:4].tolist() for row in trajectory}
    timestamps = [int(t) for t in keyframes['source_timestamps']]
    if len(timestamps) < 2 or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("need at least two ordered retained keyframes")
    if any(t not in poses for t in timestamps):
        raise ValueError("retained keyframe missing from final trajectory")
    snapshots = manifest['snapshots']
    sources = {int(s['anchor_timestamp']): set(s['source_timestamps']) for s in snapshots}
    if len(sources) != len(snapshots) or not set(sources).issubset(timestamps):
        raise ValueError("sphere anchors must be unique retained keyframes")
    nodes = [dict(timestamp=t, position=poses[t], sphere=t in sources) for t in timestamps]
    candidates, seen = [], set()
    for query in report['queries']:
        q = int(query['query_anchor'])
        for rank, candidate in enumerate(query['candidates'][:top_k], 1):
            c = int(candidate['anchor'])
            if q not in sources or c not in sources or c >= q:
                raise ValueError("candidate endpoints must be past-only sphere anchors")
            if not sources[q].isdisjoint(sources[c]):
                raise ValueError("candidate spheres share source keyframes")
            if (q, c) in seen:
                raise ValueError("duplicate candidate edge")
            seen.add((q, c))
            score = float(candidate['score'])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid BoW cosine score")
            candidates.append(dict(query=q, candidate=c, rank=rank, bow_score=score,
                                   mutual_matches=int(candidate['mutual_matches']),
                                   depth_inliers=int(candidate['geometric_inliers']),
                                   status='unverified_appearance_candidate'))
    return dict(nodes=nodes, sequential_edges=list(zip(timestamps[:-1], timestamps[1:])),
                candidate_edges=candidates, top_k=top_k, visualization_only=True,
                coordinates='Final DPVO camera-to-world positions; monocular scale, not meters',
                backbone='Consecutive retained keyframes; not the internal DPVO patch-factor graph',
                caveat='Retrieval candidates are not confirmed loop closures or inserted constraints.')


def plot_graph(graph, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    positions = np.asarray([node['position'] for node in graph['nodes']])
    xy = positions[:, [0, 2]]  # Camera-RDF world: X-Z projection, no gravity alignment.
    lookup = {node['timestamp']: point for node, point in zip(graph['nodes'], xy)}
    anchors = np.asarray([lookup[n['timestamp']] for n in graph['nodes'] if n['sphere']]).reshape(-1, 2)
    edges = graph['candidate_edges']
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
        for ax in axes:
            ax.plot(*xy.T, color='#263c55', linewidth=1.6, zorder=3)
            ax.scatter(*xy.T, s=9, color='#263c55', zorder=3)
            ax.scatter(*anchors.T, s=32, color='#268fc2', edgecolors='white', linewidths=.6, zorder=4)
            ax.scatter(*xy[0], s=90, marker='^', color='#15966b', edgecolors='white', zorder=5)
            ax.scatter(*xy[-1], s=90, marker='s', color='#9c3868', edgecolors='white', zorder=5)
            for point, label, offset in ((xy[0], 'start', (7, 6)), (xy[-1], 'end', (7, -14))):
                ax.annotate(label, point, xytext=offset, textcoords='offset points', fontsize=10)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.18)
            ax.margins(.15)
            ax.set_xlabel('X (DPVO units, not meters)')
        axes[0].set_ylabel('Z (DPVO units, not meters)')
        axes[0].set_title('Final keyframe trajectory', loc='left', fontsize=13)
        axes[1].set_title(f'{len(edges)} unverified candidate loops', loc='left', fontsize=13)
        if edges:
            segments = [[lookup[e['query']], lookup[e['candidate']]] for e in edges]
            axes[1].add_collection(LineCollection(segments, colors='#e3811c', linestyles='dashed',
                                                  linewidths=1.1, alpha=.72, zorder=2))
        handles = [Line2D([], [], color='#263c55', marker='.', label=f"{len(xy)} keyframes / sequential links"),
                   Line2D([], [], color='#268fc2', marker='o', linestyle='', label=f'{len(anchors)} sphere anchors'),
                   Line2D([], [], color='#e3811c', linestyle='--', label=f"Top-{graph['top_k']} BoW candidates (unverified)")]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .05), ncol=3, frameon=False)
        fig.suptitle(title, fontsize=17, fontweight='semibold', y=.98)
        depth_supported = sum(e['depth_inliers'] > 0 for e in edges)
        fig.text(.5, .922, f"X-Z projection | {depth_supported}/{len(edges)} candidates have depth inliers | no loop optimization applied",
                 ha='center', fontsize=11, color='#62503d')
        fig.text(.5, .025, 'Backbone connects consecutive retained keyframes; it is not the internal DPVO factor graph.',
                 ha='center', fontsize=9, color='#666666')
        fig.subplots_adjust(top=.86, bottom=.17, left=.075, right=.97, wspace=.14)
        fig.savefig(output / 'loop_graph.png', dpi=180)
        fig.savefig(output / 'loop_graph.pdf')
        plt.close(fig)


def save_rerun(graph, output):
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init('DPVO sphere candidate loop graph', strict=True)
    rr.save(str(output / 'loop_graph.rrd'))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(name='Keyframes and candidate loops', origin='world'),
        rrb.Spatial2DView(name='Trajectory graph (X-Z)', origin='plot'), column_shares=[1, 1]),
        collapse_panels=True), make_active=True, make_default=True)
    rr.log('world', rr.ViewCoordinates.RDF, static=True)
    lookup = {n['timestamp']: n['position'] for n in graph['nodes']}
    rr.log('world/sequential_links', rr.LineStrips3D([list(lookup.values())], colors=[63, 111, 170],
           radii=rr.Radius.ui_points(1.5)), static=True)
    rr.log('world/keyframes', rr.Points3D(list(lookup.values()), colors=[63, 111, 170],
           radii=rr.Radius.ui_points(2)), static=True)
    anchors = [n for n in graph['nodes'] if n['sphere']]
    if anchors:
        rr.log('world/sphere_anchors', rr.Points3D([n['position'] for n in anchors], colors=[38, 143, 194],
               labels=[str(n['timestamp']) for n in anchors], show_labels=False,
               radii=rr.Radius.ui_points(4)), static=True)
    # Rerun lines are solid: their entity names and labels explicitly say candidate.
    for edge in graph['candidate_edges']:
        path = f"world/unverified_candidates/query_{edge['query']:06d}_to_{edge['candidate']:06d}"
        rr.log(path, rr.LineStrips3D([[lookup[edge['query']], lookup[edge['candidate']]]],
               colors=[227, 129, 28], radii=rr.Radius.ui_points(1)), static=True)
        rr.log(path + '/details', rr.TextDocument(json.dumps(edge, indent=2)), static=True)
    rr.log('plot', rr.EncodedImage(path=output / 'loop_graph.png'), static=True)
    rr.log('info', rr.TextDocument(graph['backbone'] + '\n' + graph['caveat']), static=True)
    rr.get_global_data_recording().flush()
    rr.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', required=True, type=Path)
    parser.add_argument('--keyframes', required=True, type=Path, help='Saved .keyframes.json audit')
    parser.add_argument('--spheres', required=True, type=Path, help='Sphere manifest.json')
    parser.add_argument('--retrieval', required=True, type=Path, help='ORB/BoW report.json')
    parser.add_argument('--output', required=True, type=Path, help='New directory; never overwrites')
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--title', default='DPVO sphere retrieval graph')
    args = parser.parse_args()
    graph = build_graph(np.loadtxt(args.trajectory, ndmin=2), json.loads(args.keyframes.read_text()),
                        json.loads(args.spheres.read_text()), json.loads(args.retrieval.read_text()), args.top_k)
    graph['inputs'] = {key: str(value.resolve()) for key, value in vars(args).items() if isinstance(value, Path)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'loop_graph.json').write_text(json.dumps(graph, indent=2) + '\n')
    plot_graph(graph, args.output, args.title)
    save_rerun(graph, args.output)
    print(f"Saved {len(graph['nodes'])} keyframes and {len(graph['candidate_edges'])} unverified candidate edges to {args.output}")


if __name__ == '__main__':
    main()

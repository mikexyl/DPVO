"""Ablate appearance gates on frozen saved sphere rankings and descriptors."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from dpvo.sphere_bow import match_features, verify_depth_geometry
from dpvo.sphere_match_gates import GATES, match_variants, unique_targets
from inspect_sphere_matches import depth_display_masks, load_features, match_segments
from sphere_place_recognition import pair_image, read_sphere, save_rgb

DESCRIPTIONS = {
    'strict': 'Exact mutual nearest neighbors, ratio < .75 in both directions, Hamming <= 64.',
    'forward-ratio': 'Query ratio < .75 and Hamming <= 64; no reverse or mutual gate.',
    'mutual-forward-ratio': 'Exact mutual nearest neighbors and forward ratio < .75; reverse ratio disabled.',
    'patch-mutual': 'Forward ratio < .75; the reverse nearest neighbor may land within the query angular patch.',
    'patch-ratio-mutual': 'Patch mutual cycle; ratio competitor must lie outside the best candidate angular patch (closest 32 searched).',
    'distance-only': 'One-way nearest neighbors with Hamming <= 64; ratio and mutual filters disabled.',
}


def run(args):
    import rerun as rr
    import rerun.blueprint as rrb
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.retrieval/'report.json').read_text())
    root = Path(report.get('visualization_input', report['input']))
    snapshots = json.loads((root/'manifest.json').read_text())['snapshots']
    features, images = {}, {}

    def get(index):
        if index not in features:
            anchor = snapshots[index]['anchor_timestamp']
            features[index] = load_features(args.retrieval/'features'/f'{anchor:06d}.npz',
                                           report['descriptor_family'], report['descriptor_version'])
        return features[index]

    def picture(index):
        if index not in images:
            images[index] = read_sphere(root, snapshots[index], report['settings'].get('downsample', 1))[0]
        return images[index]

    rr.init('Office sphere appearance gate comparison', strict=True)
    rr.save(str(args.output/'match_gates.rrd'))
    tabs = []
    for name in GATES:
        def view(title, paths):
            return rrb.Spatial2DView(name=title, origin='pair',
                contents=['$origin/background', *[f'$origin/{name}/{p}/**' for p in paths]])
        tabs.append(rrb.Vertical(rrb.Horizontal(view('All appearance matches', ['appearance']),
                                               view('Depth: green / magenta / gray unverified', ['inliers','outliers','unverified'])),
                                 rrb.TextDocumentView(name=name, origin=f'info/{name}'),
                                 name=name, row_shares=[.85,.15]))
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs, active_tab=4), collapse_panels=True), make_active=True, make_default=True)
    records = []
    (args.output/'matches').mkdir()
    for number, query in enumerate(report['queries']):
        i, chosen = query['query_index'], query['candidates'][0]
        j = chosen['index']; qa, ca = query['query_anchor'], chosen['anchor']
        assert j < i and set(snapshots[i]['source_timestamps']).isdisjoint(snapshots[j]['source_timestamps'])
        q, c = get(i), get(j)
        start = perf_counter()
        variants = match_variants(q, c, ratio=args.ratio, patch_radius_degrees=args.patch_radius_degrees)
        matching_seconds = perf_counter() - start
        baseline = match_features(q, c)
        if args.ratio == .75:
            np.testing.assert_array_equal(variants['strict'], baseline)
        variants['strict'] = baseline  # Frozen .75 baseline even when relaxing experimental gates.
        assert len(variants['strict']) == chosen['mutual_matches']
        record = dict(query=qa, candidate=ca, matching_all_variants_seconds=matching_seconds, gates={})
        archive = {}
        # The focused pair is shown first in the recording; all queries remain
        # available by their actual sphere_anchor timestamp.
        visualize = qa == args.focus_anchor
        if visualize:
            qi, ci = picture(i), picture(j)
            h,w = qi.shape[:2]
            rr.set_time('sphere_anchor', sequence=qa)
            bg = pair_image(qi, ci, q, c, np.empty((0,2),np.int32), np.zeros(0,bool),
                            f'Query {qa} -> candidate {ca} | fixed ranking and descriptors',
                            'Compare appearance gates using the tabs; no panorama pixel-displacement constraint')
            rr.log('pair/background', rr.Image(bg))
        for name, matches in variants.items():
            geometry, inliers = verify_depth_geometry(q, c, matches, seed=7)
            if name == 'strict':
                assert geometry['geometric_inliers'] == chosen['geometric_inliers']
            unique = unique_targets(q, c, matches)
            unique_geometry, unique_inliers = verify_depth_geometry(q, c, unique, seed=7)
            stats = dict(matches=len(matches), unique_candidate_features=len(unique),
                         unique_inlier_candidate_features=len(np.unique(matches[inliers,1])),
                         depth=geometry, one_to_one_depth=unique_geometry)
            if qa in (args.focus_anchor, 336):
                stats['seed_stability'] = [dict(seed=seed,
                    all_matches=verify_depth_geometry(q,c,matches,seed=seed)[0],
                    one_to_one=verify_depth_geometry(q,c,unique,seed=seed)[0]) for seed in (7,17,27,37,47)]
            record['gates'][name] = stats
            archive[f'{name}_matches'] = matches
            archive[f'{name}_inliers'] = inliers
            archive[f'{name}_unique_matches'] = unique
            archive[f'{name}_unique_inliers'] = unique_inliers
            if visualize:
                evaluated = np.isin(matches[:,0],unique[:,0])
                display_inliers = np.zeros(len(matches),bool)
                display_inliers[evaluated] = unique_inliers
                rejected, unverified = depth_display_masks(q,c,matches,unique_geometry,display_inliers,evaluated)
                lines = match_segments(q,c,matches,w,h)
                for entity, segments, color in (
                    ('appearance',lines,[255,175,50,180]),('inliers',lines[display_inliers],[80,240,100,230]),
                    ('outliers',lines[rejected],[255,65,190,150]),('unverified',lines[unverified],[180,180,180,180])):
                    rr.log(f'pair/{name}/{entity}/lines',rr.LineStrips2D(segments,colors=color,radii=rr.Radius.ui_points(.8)))
                    rr.log(f'pair/{name}/{entity}/points',rr.Points2D(segments.reshape(-1,2),colors=color,radii=rr.Radius.ui_points(2)))
                description = DESCRIPTIONS[name] if name == 'strict' else DESCRIPTIONS[name].replace('.75',f'{args.ratio:g}')
                info = f'{name}: {description} Patch radius {args.patch_radius_degrees} degrees.\n'
                info += f'{len(matches)} appearance matches / {len(unique)} distinct candidate IDs / {int(display_inliers.sum())} depth inliers.\n'
                info += f'Depth fit keeps the lowest-Hamming match per target; status {unique_geometry["verification_status"]}. Duplicate-target matches remain visible and are gray in the depth view.\n'
                info += 'Unrestricted 3D Sim(3), 500 RANSAC trials, 3% range threshold; no known pose or panorama line direction is used. Crossing lines are allowed.\n'
                info += 'Gray is unverified. Depth consistency is diagnostic, not ground truth.\n'
                info += 'All lines shown without a cap. Other 114 frozen top-1 pairs and seed stability are in report.json.'
                rr.log(f'info/{name}',rr.TextDocument(info))
                png = pair_image(qi,ci,q,c,matches,display_inliers,f'{name}: {qa} -> {ca}',
                    f'{len(matches)} matches / {len(unique)} targets / {display_inliers.sum()} distinct-target depth inliers',unverified=unverified)
                save_rgb(args.output/f'{qa:06d}_{ca:06d}_{name}.png',png)
        np.savez_compressed(args.output/'matches'/f'{qa:06d}_{ca:06d}.npz',**archive)
        records.append(record)
        if (number+1)%10==0 or visualize:
            print(f'{number+1}/{len(report["queries"])}: {qa} -> {ca}',flush=True)
    rr.get_global_data_recording().flush();rr.disconnect()
    summaries = {}
    for name in GATES:
        data = [r['gates'][name] for r in records]
        summaries[name] = dict(matches=sum(r['matches'] for r in data),
            unique_candidate_features=sum(r['unique_candidate_features'] for r in data),
            depth_inliers=sum(r['depth']['geometric_inliers'] for r in data),
            pairs_with_depth_inliers=sum(r['depth']['geometric_inliers']>0 for r in data),
            one_to_one_depth_inliers=sum(r['one_to_one_depth']['geometric_inliers'] for r in data),
            pairs_with_one_to_one_depth_inliers=sum(r['one_to_one_depth']['geometric_inliers']>0 for r in data))
    result=dict(input=str(args.retrieval.resolve()),settings=vars(args),queries=len(records),
        visualization_depth='one_to_one_depth; omitted duplicate targets are unverified, all appearance matches remain visible',
        ranking='frozen original top-one candidates; no extraction or vocabulary training',
        descriptor_family=report['descriptor_family'],descriptor_version=report['descriptor_version'],
        descriptions={n:(d if n=='strict' else d.replace('.75',f'{args.ratio:g}')) for n,d in DESCRIPTIONS.items()},summaries=summaries,records=records,
        caveats=['Depth consensus is not ground truth.','Many-to-one matches can inflate consensus; independent-target results are reported separately.',
                 'Nearby detections across octaves can still duplicate physical structures after target-ID deduplication.',
                 '500 trials and a fixed seed are not a guarantee of the best consensus; selected pairs include five-seed stability.'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieval',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--patch-radius-degrees',type=float,default=1.);p.add_argument('--focus-anchor',type=int,default=822)
    p.add_argument('--ratio',type=float,default=.75,help='Experimental gates only; strict baseline stays at .75')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();cv2.setNumThreads(args.threads);run(args)

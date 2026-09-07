"""Compare real TEASER++ against frozen saved-sphere Sim(3) RANSAC correspondences."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from dpvo.sphere_bow import match_features, verify_depth_geometry
from dpvo.sphere_match_gates import unique_targets
from dpvo.teaser import verify_teaser_geometry
from inspect_sphere_matches import depth_display_masks, load_features, match_segments
from sphere_place_recognition import pair_image, read_sphere, save_rgb


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'matches').mkdir()
    start = perf_counter()
    from dpvo import _teaser  # Separate setup/import from warmed computation.
    setup_seconds = perf_counter() - start
    report = json.loads((args.retrieval/'report.json').read_text())
    prepare_gates = args.gates is None
    if prepare_gates:
        args.gates = args.output/'appearance_gates'
        (args.gates/'matches').mkdir(parents=True)
        gates = dict(input=str(args.retrieval.resolve()),settings=dict(ratio=.8,patch_radius_degrees=1.),records=[])
    else:
        gates = json.loads((args.gates/'report.json').read_text())
    if gates['settings']['ratio'] != .8:
        raise ValueError('This comparison requires the saved ratio=.80 appearance gates')
    root = Path(report.get('visualization_input', report['input']))
    snapshots = json.loads((root/'manifest.json').read_text())['snapshots']
    frozen = {(r['query'],r['candidate']):r for r in gates['records']}
    features, records, cached = {}, [], []
    preparation_seconds = 0.
    io_start = perf_counter()
    for query in report['queries']:
        i, chosen = query['query_index'], query['candidates'][0]
        j, qa, ca = chosen['index'], query['query_anchor'], chosen['anchor']
        assert j < i and set(snapshots[i]['source_timestamps']).isdisjoint(snapshots[j]['source_timestamps'])
        for index, anchor in ((i,qa),(j,ca)):
            if index not in features:
                features[index] = load_features(args.retrieval/'features'/f'{anchor:06d}.npz',
                    report['descriptor_family'], report['descriptor_version'])
                f = features[index]
                valid = np.isfinite(f.points).all(axis=1)
                # Assert the stored points really are spherical radial-depth points.
                np.testing.assert_allclose(f.points[valid], f.bearings[valid] * np.linalg.norm(f.points[valid],axis=1)[:,None],atol=2e-6)
        q,c = features[i],features[j]
        archive_path = args.gates/'matches'/f'{qa:06d}_{ca:06d}.npz'
        if prepare_gates:
            start = perf_counter()
            strict = match_features(q,c)
            assert len(strict) == chosen['mutual_matches']
            raw = match_features(q,c,ratio=.8,gate='patch-ratio-mutual')
            matches = unique_targets(q,c,raw)
            geometry,baseline = verify_depth_geometry(q,c,matches,seed=7)
            preparation_seconds += perf_counter() - start
            row = dict(query=qa,candidate=ca,strict_matches=len(strict),gates={
                'patch-ratio-mutual':dict(matches=len(raw),unique_candidate_features=len(matches),one_to_one_depth=geometry)})
            gates['records'].append(row)
            frozen[(qa,ca)] = row
            np.savez_compressed(archive_path,**{'patch-ratio-mutual_matches':raw,
                'patch-ratio-mutual_unique_matches':matches,'patch-ratio-mutual_unique_inliers':baseline})
        else:
            with np.load(archive_path) as archive:
                raw = archive['patch-ratio-mutual_matches'].copy()
                matches = archive['patch-ratio-mutual_unique_matches'].copy()
                baseline = archive['patch-ratio-mutual_unique_inliers'].copy()
        assert len(np.unique(matches[:,0])) == len(matches) == len(np.unique(matches[:,1]))
        cached.append((query,q,c,raw,matches,baseline))
        records.append(dict(query=qa,candidate=ca,appearance_matches=len(raw),unique_target_matches=len(matches),
                            correspondence_sha256=hashlib.sha256(matches.tobytes()).hexdigest(),methods={}))
    io_seconds = perf_counter() - io_start - preparation_seconds
    if prepare_gates:
        (args.gates/'report.json').write_text(json.dumps(gates,indent=2)+'\n')
        print(f'Prepared and froze patch-ratio-mutual .80 correspondences for {len(records)} pairs',flush=True)
    if not records:
        raise ValueError('no eligible saved query pairs')
    anchors = args.focus_anchors
    if anchors is None:
        # Display choice uses appearance evidence only, never geometric success.
        anchors = [r['query'] for r in sorted(records,key=lambda r:(-r['unique_target_matches'],-r['appearance_matches'],r['query']))[:2]]
    if len(set(anchors)) != len(anchors) or not set(anchors).issubset(r['query'] for r in records):
        raise ValueError('focus anchors must be distinct eligible query anchors')
    args.focus_anchors = anchors
    methods = ['ransac'] + [f'teaser-{f:g}' for f in args.noise_fractions]
    timings = {name:[] for name in methods}
    native_timings = {name:[] for name in methods}
    called_timings = {name:[] for name in methods}
    overhead_timings = {name:[] for name in methods}
    # One warm-up pass plus three measured passes. No image/file I/O in these loops.
    for pass_index in range(args.passes + 1):
        for item, record in zip(cached, records):
            query,q,c,raw,matches,baseline = item
            for name in methods:
                start = perf_counter()
                if name == 'ransac':
                    geometry, inliers = verify_depth_geometry(q,c,matches,seed=7)
                else:
                    geometry, inliers = verify_teaser_geometry(q,c,matches,noise_fraction=float(name.split('-')[1]),workers=args.workers)
                elapsed = perf_counter() - start
                if name == 'ransac':
                    np.testing.assert_array_equal(inliers, baseline)
                    assert geometry == frozen[(record['query'],record['candidate'])]['gates']['patch-ratio-mutual']['one_to_one_depth']
                if pass_index:
                    timings[name].append(elapsed)
                    native = geometry.get('solver',{}).get('native_seconds')
                    if native is not None:
                        native_timings[name].append(native)
                        called_timings[name].append(elapsed)
                        overhead_timings[name].append(elapsed-native)
                previous = record['methods'].get(name)
                if pass_index == 0:
                    record['methods'][name] = dict(geometry=geometry,inliers=inliers.tolist(),
                        warmed_inlier_counts=[],warmed_scales=[],mask_changes=0)
                else:
                    previous['warmed_inlier_counts'].append(int(inliers.sum()))
                    previous['warmed_scales'].append(geometry['scale'])
                    previous['mask_changes'] += int(not np.array_equal(previous['inliers'],inliers))
        print(f'Finished {"warm-up" if pass_index == 0 else f"measured pass {pass_index}"}: {len(records)} frozen pairs',flush=True)
    for record, item in zip(records,cached):
        query,q,c,raw,matches,_ = item
        i,j=query['query_index'],query['candidates'][0]['index']
        expected=np.asarray(snapshots[j]['camera_to_world_rotation']).T @ np.asarray(snapshots[i]['camera_to_world_rotation'])
        def angle(rotation):
            return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))
        record['saved_pose_audit_only']=dict(relative_rotation_degrees=angle(expected),
            query_negative_z_points=int((q.points[matches[:,0],2]<0).sum()),
            candidate_negative_z_points=int((c.points[matches[:,1],2]<0).sum()))
        for name in methods[1:]:
            g=record['methods'][name]['geometry']
            if g['verification_status']=='checked':
                g['rotation_degrees']=angle(np.asarray(g['rotation']))
                g['rotation_error_vs_saved_pose_degrees']=angle(np.asarray(g['rotation']) @ expected.T)
        np.savez_compressed(args.output/'matches'/f'{record["query"]:06d}_{record["candidate"]:06d}.npz',
                            appearance_matches=raw,unique_matches=matches,
                            **{name:np.asarray(record['methods'][name]['inliers'],bool) for name in methods})
    def percentiles(values):
        return dict(median_ms=float(np.median(values)*1000),p95_ms=float(np.percentile(values,95)*1000),samples=len(values)) if values else None
    summaries = {}
    for name in methods:
        rows = [r['methods'][name] for r in records]
        summaries[name] = dict(depth_inliers=sum(r['geometry']['geometric_inliers'] for r in rows),
            supported_pairs=sum(r['geometry']['geometric_inliers']>=6 for r in rows),
            warmed_total_inliers=[sum(r['warmed_inlier_counts'][i] for r in rows) for i in range(args.passes)],
            pairs_with_changing_masks=sum(r['mask_changes']>0 for r in rows),
            warmed_verification=percentiles(timings[name]),warmed_native_when_called=percentiles(native_timings[name]),
            warmed_verification_when_native_called=percentiles(called_timings[name]),
            warmed_wrapper_when_native_called=percentiles(overhead_timings[name]))
    result = dict(retrieval=str(args.retrieval.resolve()),gate_comparison=str(args.gates.resolve()),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        queries=len(records),appearance_matches=sum(r['appearance_matches'] for r in records),
        descriptor_family=report['descriptor_family'],descriptor_version=report['descriptor_version'],
        unique_target_matches=sum(r['unique_target_matches'] for r in records),
        setup_import_seconds=setup_seconds,feature_and_match_io_seconds=io_seconds,
        appearance_and_baseline_preparation_seconds=preparation_seconds,
        camera_model='none; stored RDF unit bearing times radial depth; unrestricted proper 3D rotation',
        noise_model='absolute isotropic bound = noise_fraction * median target range; common per-point 3% range gate afterward',
        solver='Pinned upstream TEASER++ TLS scale, PMC_EXACT with 2-second search limit, CHAIN GNC-TLS rotation, TLS translation; no pose prior or least-squares post-refit',
        caveats=['All edges remain candidates; depth consistency is not ground truth.',
                 'RANSAC input and output masks are checked against the frozen baseline.',
                 'TEASER fitting omits exact coincident positions to avoid zero-length scale measurements; the common gate still checks all unique-target correspondences.',
                 'PMC may choose different equal cliques across threads; the upstream API exposes no timeout/optimality flag. No certification is claimed.',
                 'Noise sweep results are exploratory; a larger depth-consensus count does not prove more correct place matches.'],
        summaries=summaries,records=records)
    (args.output/'report.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    viz_start = perf_counter()
    export(args,report,root,snapshots,cached,records,methods)
    result['visualization_io_and_export_seconds'] = perf_counter() - viz_start
    (args.output/'report.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


def export(args,report,root,snapshots,cached,records,methods):
    import rerun as rr
    import rerun.blueprint as rrb
    rr.init(Path(report['input']).name+'_teaser_sim3_sphere_comparison',strict=True)
    rr.save(str(args.output/'teaser_comparison.rrd'))
    tabs=[]
    for name in methods:
        def view(title,paths):
            return rrb.Spatial2DView(name=title,origin='pair',contents=['$origin/background',*[f'$origin/{p}/**' for p in paths]])
        tabs.append(rrb.Vertical(rrb.Horizontal(view('All appearance matches',['appearance']),
            view(name+': green depth support / magenta rejected / gray unverified',[f'{name}/inliers',f'{name}/outliers',f'{name}/unverified'])),
            rrb.TextDocumentView(name=name,origin=f'info/{name}'),name=name,row_shares=[.83,.17]))
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs,active_tab=methods.index('teaser-0.03') if 'teaser-0.03' in methods else 1),collapse_panels=True),make_active=True,make_default=True)
    priority={anchor:i for i,anchor in enumerate(args.focus_anchors)}
    order=sorted(range(len(records)),key=lambda k:(priority.get(records[k]['query'],len(priority)),k))
    slider_hint='; '.join(f'{i} = {records[k]["query"]} -> {records[k]["candidate"]}' for i,k in enumerate(order[:2]))
    for frame,k in enumerate(order):
        query,q,c,raw,matches,_=cached[k];record=records[k]
        qa,ca=record['query'],record['candidate'];i=query['query_index'];j=query['candidates'][0]['index']
        qi=read_sphere(root,snapshots[i],report['settings'].get('downsample',1))[0]
        ci=read_sphere(root,snapshots[j],report['settings'].get('downsample',1))[0]
        h,w=qi.shape[:2]
        rr.set_time('pair',sequence=frame)
        bg=pair_image(qi,ci,q,c,np.empty((0,2),np.int32),np.zeros(0,bool),
            f'Query {qa} -> candidate {ca} | frozen patch-ratio-mutual .80',
            'Compare solver tabs. No camera projection or panorama line-direction gate. All edges are candidates.')
        rr.log('pair/background',rr.Image(bg).compress(jpeg_quality=90))
        lines=match_segments(q,c,raw,w,h)
        def log(entity,segments,color):
            rr.log(entity+'/lines',rr.LineStrips2D(segments,colors=color,radii=rr.Radius.ui_points(.8)))
            rr.log(entity+'/points',rr.Points2D(segments.reshape(-1,2),colors=color,radii=rr.Radius.ui_points(2)))
        log('pair/appearance',lines,[255,175,50,180])
        evaluated=np.isin(raw[:,0],matches[:,0])
        for name in methods:
            g=record['methods'][name]['geometry'];mask=np.zeros(len(raw),bool)
            mask[evaluated]=record['methods'][name]['inliers']
            rejected,unverified=depth_display_masks(q,c,raw,g,mask,evaluated)
            for entity,selection,color in [('inliers',mask,[80,240,100,230]),('outliers',rejected,[255,65,190,150]),('unverified',unverified,[180,180,180,180])]:
                log(f'pair/{name}/{entity}',lines[selection],color)
            info=f'{qa} -> {ca}: {len(raw)} appearance matches / {len(matches)} distinct targets / {mask.sum()} depth-supporting matches.\n'
            info+=f'{name}: {g["verification_status"]}; scale={g["scale"]}. Final gate: 3% of each target range; scale must be .25..4.\n'
            if name!='ransac':
                solver=g.get('solver',{})
                info+=f'Upstream C++ TEASER++; absolute noise bound {solver.get("noise_bound")}; clique {len(solver.get("clique_indices",[]))}. No certification claimed.\n'
                if 'rotation_error_vs_saved_pose_degrees' in g:
                    info+=f'Rotation differs from saved poses by {g["rotation_error_vs_saved_pose_degrees"]:.2f} degrees (audit only; poses were not used to fit or gate).\n'
            info+='Fit uses bearing x radial-depth 3D points. Rear hemisphere, poles, seams and crossing lines are allowed.\n'
            info+='Gray means no accepted geometric model, missing depth, or duplicate target omitted from fitting; it does not mean an appearance mismatch.\n'
            info+='All appearance lines are shown. Depth consistency is diagnostic; all edges remain candidates. Pair slider: '+slider_hint+'.'
            rr.log(f'info/{name}',rr.TextDocument(info))
            if qa in args.focus_anchors and name in ('ransac','teaser-0.03'):
                save_rgb(args.output/f'{qa:06d}_{ca:06d}_{name}.png',pair_image(qi,ci,q,c,raw,mask,
                    f'{name}: {qa} -> {ca}',f'{len(raw)} appearance / {len(matches)} targets / {mask.sum()} depth support; {g["verification_status"]}',unverified=unverified))
    rr.get_global_data_recording().flush();rr.disconnect()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--gates',type=Path,help='Reuse a frozen ratio=.80 gate comparison; otherwise prepare from saved features')
    p.add_argument('--focus-anchors',type=int,nargs='*',help='Display these query anchors first; default: two with most distinct appearance targets')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--noise-fractions',nargs='+',type=float,default=[.01,.02,.03,.05,.1])
    p.add_argument('--passes',type=int,default=3)
    args=p.parse_args()
    if args.passes<1 or not args.noise_fractions or len(set(args.noise_fractions))!=len(args.noise_fractions):p.error('positive passes and distinct noise fractions required')
    cv2.setNumThreads(args.workers);run(args)

"""Show virtual-SPHORB matches on the original source keyframe photographs."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from dpvo.sphere_bow import match_features, verify_depth_geometry
from inspect_sphere_matches import depth_display_masks, load_features


def export(experiment, output, match_gate=None, patch_radius_degrees=None, query_anchor=None, match_ratio=None, depth_unique_targets=False):
    import rerun as rr
    import rerun.blueprint as rrb
    report=json.loads((experiment/'retrieval/report.json').read_text())
    source=json.loads((experiment/'virtual_input.json').read_text())
    override=match_gate is not None or patch_radius_degrees is not None or match_ratio is not None or depth_unique_targets
    gate=match_gate or report['settings'].get('match_gate','strict')
    radius=patch_radius_degrees if patch_radius_degrees is not None else report['settings'].get('patch_radius_degrees',1.)
    ratio=match_ratio if match_ratio is not None else report['settings'].get('match_ratio',.75)
    cache=Path(report['settings']['source_cache']).resolve()
    output.mkdir(parents=True,exist_ok=False)
    rr.init('Virtual SPHORB: original image correspondences',strict=True)
    rr.save(str(output/'source_image_matches.rrd'))
    def view(name,entities):
        return rrb.Spatial2DView(name=name,origin='pair',contents=['$origin/background',*[f'$origin/{e}/**' for e in entities]])
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(view('All appearance matches on original images',['appearance']),
                       view('Depth: green inliers / magenta outliers / gray unverified',['inliers','outliers','unverified'])),
        rrb.TextDocumentView(name='Sphere anchors and exact source frames',origin='info'),row_shares=[.86,.14]),
        collapse_panels=True),make_active=True,make_default=True)

    @lru_cache(maxsize=60)
    def image(timestamp):
        bgr=cv2.imread(str(cache/f'{timestamp:06d}.png'))
        if bgr is None:raise ValueError(f'missing original image {timestamp}')
        h,w=bgr.shape[:2]
        rgb=cv2.cvtColor(cv2.resize(bgr,(960,round(h*960/w)),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        return rgb,np.array(rgb.shape[1::-1])/np.array([w,h])

    @lru_cache(maxsize=None)
    def features(anchor):
        f=load_features(experiment/'features'/f'{anchor:06d}.npz',report['descriptor_family'],report['descriptor_version'])
        with np.load(experiment/'provenance'/f'{anchor:06d}.npz') as p:
            prov={k:p[k].copy() for k in p.files}
        if len(prov['source_uv'])!=len(f.uv):raise ValueError('feature/source provenance length mismatch')
        return f,prov

    records=[]
    for query in report['queries']:
        if query_anchor is not None and query['query_anchor'] != query_anchor:continue
        qanchor=query['query_anchor'];selected=query['candidates'][0];canchor=selected['anchor']
        q,qp=features(qanchor);c,cp=features(canchor)
        matches=match_features(q,c,ratio=ratio,gate=gate,patch_radius_degrees=radius)
        evaluated=np.ones(len(matches),bool)
        depth_matches=matches
        if depth_unique_targets:
            from dpvo.sphere_match_gates import unique_targets
            depth_matches=unique_targets(q,c,matches)
            evaluated=np.isin(matches[:,0],depth_matches[:,0])
        geometry,depth_inliers=verify_depth_geometry(q,c,depth_matches,seed=report['settings']['seed'])
        inliers=np.zeros(len(matches),bool);inliers[evaluated]=depth_inliers
        rejected,unverified=depth_display_masks(q,c,matches,geometry,inliers,evaluated)
        if not override and (len(matches)!=selected['mutual_matches'] or int(inliers.sum())!=selected['geometric_inliers']):
            raise ValueError('matches changed since retrieval')
        if not len(matches):continue
        groups=np.column_stack((qp['source_timestamps'][matches[:,0]],cp['source_timestamps'][matches[:,1]]))
        for qsource,csource in np.unique(groups,axis=0):
            indices=np.flatnonzero((groups==[qsource,csource]).all(axis=1))
            pair=matches[indices];good=inliers[indices]
            qi,qs=image(int(qsource));ci,cs=image(int(csource))
            canvas=np.full((qi.shape[0]+ci.shape[0]+72,960,3),28,np.uint8)
            canvas[36:36+qi.shape[0]]=qi;canvas[72+qi.shape[0]:]=ci
            for y,text in ((24,f'Query sphere {qanchor} | original keyframe {qsource}'),
                           (qi.shape[0]+60,f'Candidate sphere {canchor} | original keyframe {csource}')):
                cv2.putText(canvas,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
            a=(qp['source_uv'][pair[:,0]]+.5)*qs-.5+[0,36]
            b=(cp['source_uv'][pair[:,1]]+.5)*cs-.5+[0,qi.shape[0]+72]
            lines=np.stack((a,b),axis=1).astype(np.float32)
            rr.set_time('source_pair',sequence=len(records))
            rr.log('pair/background',rr.Image(canvas))
            for name,segments,color in (('appearance',lines,[255,170,50]),('inliers',lines[good],[80,240,100]),
                                        ('outliers',lines[rejected[indices]],[255,65,190]),
                                        ('unverified',lines[unverified[indices]],[180,180,180])):
                rr.log(f'pair/{name}/lines',rr.LineStrips2D(segments,colors=color,radii=rr.Radius.ui_points(1)))
                rr.log(f'pair/{name}/points',rr.Points2D(segments.reshape(-1,2),colors=color,radii=rr.Radius.ui_points(3)))
            image_input=source['source'].get('source_kind')=='images'
            raw=lambda t:source['source']['skip']+(int(t) if image_input else int(t)+1)*source['source']['stride']-(0 if image_input else 1)
            text=f'Spheres {qanchor} -> {canchor}; source keyframes {qsource} -> {csource}; raw source indices {raw(qsource)} -> {raw(csource)}.\n'
            text+=f'Appearance gate: {gate}; ratio {ratio}; angular patch radius {radius} degrees. Saved retrieval ranking is unchanged.\n'
            if depth_unique_targets:text+=f'Depth fit uses {len(depth_matches)} unique target IDs from {len(matches)} appearance matches; duplicate-target matches are gray in the depth view.\n'
            text+=f'All {len(pair)} appearance matches from this source pair; {int(good.sum())} depth inliers. Orange includes every appearance match, without a display cap.\n'
            text+=f'Depth status: {geometry["verification_status"]}; {int(rejected[indices].sum())} tested outliers, {int(unverified[indices].sum())} unverified (gray). At least 6 valid-depth matches are required to fit unrestricted 3D rotation, translation and scale.\n'
            text+='The points mark fractional pixel centers in the original photographs. SPHORB descriptors use source-image samples warped onto the geodesic grid, with full support from one source.\n'
            text+='Depth consistency is diagnostic, not ground truth. Scrub source_pair to inspect every source-image group from each top-ranked candidate.'
            rr.log('info',rr.TextDocument(text))
            records.append(dict(query_anchor=qanchor,candidate_anchor=canchor,query_source=int(qsource),candidate_source=int(csource),
                                appearance_matches=len(pair),depth_inliers=int(good.sum()),
                                rejected_by_depth=int(rejected[indices].sum()),not_verified=int(unverified[indices].sum()),
                                verification_status=geometry['verification_status']))
    rr.get_global_data_recording().flush();rr.disconnect()
    (output/'report.json').write_text(json.dumps(dict(groups=records,displayed_match_cap=None,match_gate=gate,
        patch_radius_degrees=radius,match_ratio=ratio,depth_unique_targets=depth_unique_targets,
        ranking='saved top-one candidate, unchanged'),indent=2)+'\n')
    print(f'Saved {len(records)} original-image groups: {output/"source_image_matches.rrd"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    from dpvo.sphere_match_gates import GATES
    p.add_argument('--match-gate',choices=GATES);p.add_argument('--patch-radius-degrees',type=float)
    p.add_argument('--query-anchor',type=int)
    p.add_argument('--match-ratio',type=float)
    p.add_argument('--depth-unique-targets',action='store_true',help='Fit depth only after retaining the lowest-Hamming match per candidate ID; keep every appearance match visible')
    args=p.parse_args();cv2.setNumThreads(4);export(args.experiment.resolve(),args.output.resolve(),args.match_gate,args.patch_radius_degrees,args.query_anchor,args.match_ratio,args.depth_unique_targets)

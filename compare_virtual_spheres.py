"""Compare a completed virtual-source run with saved panorama-input runs."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dpvo.sphere_bow import match_features, panorama_uv, verify_depth_geometry
from inspect_sphere_matches import load_features
from sphere_place_recognition import feature_image, pair_image, read_sphere, save_rgb


def aggregate(report):
    out=dict(report['summary'])
    for k in (1,3):
        candidates=[c for q in report['queries'] for c in q['candidates'][:k]]
        out.update({f'top{k}_appearance_matches':sum(c['mutual_matches'] for c in candidates),
                    f'top{k}_depth_inliers':sum(c['geometric_inliers'] for c in candidates),
                    f'top{k}_candidates_with_depth_inliers':sum(c['geometric_inliers']>0 for c in candidates)})
    out['mean_occupied_angular_bins']=float(np.mean([s['angular_distribution']['occupied_bins'] for s in report['features']]))
    return out


def compare(experiment, baselines, overlay_anchors=None, reference_pair=None):
    paths=dict(baselines,virtual=experiment/'retrieval')
    reports={name:json.loads((path/'report.json').read_text()) for name,path in paths.items()}
    # Older prepared-feature reports measured only the preview read here.
    # Include the recorded extraction duration, while keeping original source
    # I/O, mesh building, and provenance writes in their separate timing fields.
    virtual=reports['virtual']
    for row in virtual['features']:
        row['load_and_extract_seconds']=row['file_io_seconds']+row['preprocessing_seconds']+row['extraction_seconds']
    virtual['summary']['mean_load_extract_ms']=float(np.mean([row['load_and_extract_seconds'] for row in virtual['features']])*1000)
    (paths['virtual']/'report.json').write_text(json.dumps(virtual,indent=2)+'\n')
    summaries={name:aggregate(report) for name,report in reports.items()}
    inputs=json.loads((experiment/'virtual_input.json').read_text())
    root=Path(reports['virtual']['input'])
    snapshots=json.loads((root/'manifest.json').read_text())['snapshots']
    snapshots={s['anchor_timestamp']:s for s in snapshots}
    features={}
    def get(name,anchor):
        if (name,anchor) not in features:
            r=reports[name]
            features[name,anchor]=load_features(paths[name]/'features'/f'{anchor:06d}.npz',r['descriptor_family'],r['descriptor_version'])
        return features[name,anchor]
    virtual_data=np.load(paths['virtual']/'retrieval.npz')
    validation={}
    for name,path in baselines.items():
        with np.load(path/'retrieval.npz') as data:
            validation[name]=dict(anchors_identical=bool(np.array_equal(data['anchors'],virtual_data['anchors'])),
                                  candidate_eligibility_identical=bool(np.array_equal(data['eligible'],virtual_data['eligible'])),
                                  training_anchors_identical=reports[name]['training']==reports['virtual']['training'])
    for anchor in virtual_data['anchors']:
        f=get('virtual',int(anchor))
        expected=panorama_uv(f.bearings.astype(float),1024,512)
        delta=f.uv-expected;delta[:,0]=(delta[:,0]+512)%1024-512
        np.testing.assert_allclose(delta,0,atol=2e-4)
        np.testing.assert_allclose(np.linalg.norm(f.bearings,axis=1),1,atol=3e-7)
        with np.load(experiment/'provenance'/f'{anchor:06d}.npz') as p:
            assert len(p['source_uv'])==len(f.uv) and np.isfinite(p['source_uv']).all()
            assert set(p['source_timestamps']).issubset(snapshots[int(anchor)]['source_timestamps'])
            assert (p['source_uv']>=0).all() and (p['source_uv']<np.array(inputs['source']['raw_size'])-1).all()
    indexed={name:{(q['query_anchor'],c['anchor']):c for q in report['queries'] for c in q['candidates']}
             for name,report in reports.items()}
    common={}
    fixed={}
    for name in baselines:
        overlap=indexed[name].keys()&indexed['virtual'].keys()
        common[name]={n:dict(pairs=len(overlap),appearance_matches=sum(indexed[n][pair]['mutual_matches'] for pair in overlap),
                            depth_inliers=sum(indexed[n][pair]['geometric_inliers'] for pair in overlap)) for n in (name,'virtual')}
        pairs=[]
        for q in reports[name]['queries']:
            c=q['candidates'][0];a,b=q['query_anchor'],c['anchor']
            matches=match_features(get('virtual',a),get('virtual',b))
            geometry,_=verify_depth_geometry(get('virtual',a),get('virtual',b),matches,seed=7)
            pairs.append(dict(query=a,candidate=b,baseline_matches=c['mutual_matches'],baseline_inliers=c['geometric_inliers'],
                              virtual_matches=len(matches),virtual_inliers=geometry['geometric_inliers']))
        fixed[name]=dict(pairs=pairs,baseline_matches=sum(p['baseline_matches'] for p in pairs),
                         baseline_inliers=sum(p['baseline_inliers'] for p in pairs),
                         virtual_matches=sum(p['virtual_matches'] for p in pairs),virtual_inliers=sum(p['virtual_inliers'] for p in pairs))
    def picture(name,anchor):
        r=reports[name];image_root=Path(r.get('visualization_input',r['input']))
        return read_sphere(image_root,snapshots[anchor],r['settings'].get('downsample',1))[0]
    available=list(map(int,virtual_data['anchors']))
    if overlay_anchors is None:
        overlay_anchors=[available[i] for i in np.linspace(0,len(available)-1,min(6,len(available)),dtype=int)]
    if not set(overlay_anchors).issubset(available):
        raise ValueError('overlay anchors must belong to this sequence')
    for anchor in overlay_anchors:
        panels=[]
        for name in paths:
            f=get(name,anchor);image=feature_image(picture(name,anchor),f)
            image=cv2.resize(image,(1024,512),interpolation=cv2.INTER_NEAREST)
            image=cv2.copyMakeBorder(image,36,0,0,0,cv2.BORDER_CONSTANT,value=(28,28,28))
            cv2.putText(image,f'{name} | anchor {anchor} | {len(f.uv)} features',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            panels.append(image)
        save_rgb(experiment/f'comparison_{anchor:06d}.png',np.concatenate(panels,axis=1))
    if reference_pair is None:
        _,qa,ca=max((q['candidates'][0]['mutual_matches'],q['query_anchor'],q['candidates'][0]['anchor']) for q in virtual['queries'])
    else:
        qa,ca=reference_pair
    panels=[];reference_methods={}
    for name in paths:
        a,b=get(name,qa),get(name,ca);matches=match_features(a,b);geometry,_=verify_depth_geometry(a,b,matches,seed=7)
        reference_methods[name]=dict(appearance_matches=len(matches),**geometry)
        image=pair_image(picture(name,qa),picture(name,ca),a,b,matches,np.zeros(len(matches),bool),
                         f'{name}: {qa} -> {ca} | all {len(matches)} appearance matches',f'{geometry["geometric_inliers"]} depth inliers; all appearance matches shown')
        panels.append(cv2.resize(image,(1024,1096),interpolation=cv2.INTER_NEAREST))
    save_rgb(experiment/f'comparison_matches_{qa:06d}_{ca:06d}.png',np.concatenate(panels,axis=1))
    timing={}
    for key in ('extraction_seconds','scene_seconds','source_io_seconds','visibility_seconds','raytrace_seconds',
                'texture_seconds','detector_native_seconds','provenance_write_seconds','preview_seconds'):
        values=[r[key] for r in inputs['records']]
        timing[key]=dict(median=float(np.median(values)),p95=float(np.percentile(values,95)))
    timing['warmed_anchors']=inputs['warmed']
    result=dict(summaries=summaries,validation=validation,common_ranked_pairs=common,fixed_baseline_top1_pairs=fixed,
                reference_pair=dict(query=qa,candidate=ca,methods=reference_methods),virtual_timing=timing,
                note='Separate vocabularies on the initial training spheres; actual vocabulary sizes are reported in summaries. Depth inlier totals are diagnostics on noisy DA3 geometry; no ground-truth loop labels. Virtual previews are for display only. Baseline timings come from previous runs.')
    (experiment/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summaries=summaries,validation=validation,common_ranked_pairs=common,reference_pair=result['reference_pair'],
                         virtual_timing={k:v for k,v in timing.items() if k!='warmed_anchors'}),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--baseline',action='append',required=True,help='NAME=RETRIEVAL_DIRECTORY')
    p.add_argument('--overlay-anchors',type=int,nargs='+')
    p.add_argument('--reference-pair',type=int,nargs=2,metavar=('QUERY','CANDIDATE'))
    args=p.parse_args();cv2.setNumThreads(4)
    compare(args.experiment.resolve(),{s.split('=',1)[0]:Path(s.split('=',1)[1]).resolve() for s in args.baseline},args.overlay_anchors,args.reference_pair)

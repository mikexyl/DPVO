"""Add ScaleMaster's refined reference to an existing offline sphere pose graph."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dpvo.sphere_pose_graph import Graph
from dpvo.trajectory_reference import (align_positions, apply_alignment, position_errors,
                                       video_reference)
from optimize_sphere_pose_graph import plots, read_json, recording


def plot_reference(output, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt = reference['positions']
    report = reference['report']
    fig,axes=plt.subplots(1,3,figsize=(16,6),sharex=True,sharey=True)
    series=[('Original DPVO',reference['original'], '#778391'),
            ('Sphere-loop Sim3 PGO',reference['optimized'], '#197dcc')]
    for ax,(name,points,color),key in zip(axes[:2],series,['original','optimized']):
        ax.plot(gt[:,0],gt[:,2],color='#28a34d',linewidth=2,label='Refined reference')
        ax.plot(points[:,0],points[:,2],color=color,linewidth=1.5,label=name)
        ax.set_title(f'{name}\nATE RMSE {report["independent_sim3"][key]["rmse_m"]:.3f} m')
    axes[2].plot(gt[:,0],gt[:,2],color='#28a34d',linewidth=2,label='Refined reference')
    for name,points,color in series:
        axes[2].plot(points[:,0],points[:,2],color=color,linewidth=1.3,label=name)
    axes[2].set_title('Both aligned trajectories')
    for ax in axes:
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend(fontsize=9)
        ax.set_xlabel('ARKit world X (m)');ax.set_ylabel('ARKit world Z (m)')
    fig.suptitle(f'Office | refined ground-truth reference | {len(gt)} matched keyframes',y=.965)
    fig.text(.5,.025,'One independent global Sim3 fit per trajectory, using the same exact video frames. Reference: optimized_odometry.csv.',
             ha='center',fontsize=10)
    fig.subplots_adjust(left=.065,right=.98,bottom=.15,top=.84,wspace=.25)
    fig.savefig(output/'reference_comparison.png',dpi=180);fig.savefig(output/'reference_comparison.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--source-cache',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    summary=read_json(args.graph/'report.json');edges=read_json(args.graph/'edges.json')
    for value in summary['inputs'].values():
        if hashlib.sha256(Path(value['path']).read_bytes()).hexdigest()!=value['sha256']:
            raise ValueError('pose graph input provenance changed')
    with np.load(args.graph/'graph.npz') as data:
        graph=Graph(*[data[k].copy() for k in ('initial','source','target','measurements','sigmas','loops','confidence')])
        optimized=data['optimized'].copy();timestamps=data['timestamps'].copy()
    cache=read_json(args.source_cache/'manifest.json')
    if cache.get('source_kind')=='images' or 'video' not in cache:
        raise ValueError('this reference association requires a saved video stream')
    if cache['trajectory_sha256']!=summary['inputs']['trajectory']['sha256']:
        raise ValueError('source cache belongs to a different trajectory')
    reference=video_reference(args.reference,timestamps,stride=cache['stride'],skip=cache['skip'])
    for timestamp,raw_id in zip(timestamps,reference['raw_frame_ids']):
        saved=read_json(args.source_cache/f'{timestamp:06d}.json')
        if saved['raw_frame']!=raw_id or not saved['source_color_exact']:
            raise ValueError('frame association disagrees with validated original source pixels')
    metrics={}
    for key,poses in [('original',graph.initial),('optimized',optimized)]:
        reference[key],alignment=align_positions(poses[:,:3],reference['positions'])
        reference[key+'_errors'],error=position_errors(reference[key],reference['positions'])
        metrics[key]=dict(**error,alignment=alignment)
    reference['shared_original']=reference['original'].copy()
    reference['shared_optimized']=apply_alignment(optimized[:,:3],metrics['original']['alignment'])
    _,shared_error=position_errors(reference['shared_optimized'],reference['positions'])
    report=dict(reference=str(args.reference.resolve()),reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        reference_description='ScaleMaster SE3-refined ARKit trajectory: optimized_odometry.csv; dataset-provided reference, not independent motion capture',
        dataset_documentation='https://github.com/JooHyoSeok/ScaleMaster-Dataset#-pose-refinement-pipeline',
        coordinates='ARKit world, Y-up, positions in meters; camera positions only, no camera projection or orientation scoring',
        association=dict(formula='raw frame = skip + (DPVO timestamp + 1) * stride - 1',
                         stride=cache['stride'],skip=cache['skip'],matched_keyframes=len(timestamps),
                         reference_frame_count=reference['reference_frame_count'],first_raw_frame=int(reference['raw_frame_ids'][0]),
                         last_raw_frame=int(reference['raw_frame_ids'][-1]),
                         all_source_pixel_frame_audits_match=True,no_interpolation=True),
        independent_sim3=metrics,shared_alignment=dict(fitted_to='original DPVO only',optimized_errors=shared_error),
        protocol=f'Each trajectory receives one global Umeyama Sim3 fit against all identical {len(timestamps)} matched keyframes; equal keyframe weighting; no local or segment alignment',
        pose_graph_inputs_unchanged=True,optimization_rerun=False,
        inputs={name:dict(path=str((args.graph/name).resolve()),sha256=hashlib.sha256((args.graph/name).read_bytes()).hexdigest())
                for name in ['graph.npz','report.json','edges.json']})
    reference['report']=report
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'reference_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'reference.npz',timestamps=timestamps,
                        **{k:v for k,v in reference.items() if isinstance(v,np.ndarray)})
    np.savetxt(args.output/'matched_reference.tum',np.c_[reference['seconds'],reference['positions'],reference['quaternions']],
               fmt='%.12g',header='reference timestamp seconds tx ty tz qx qy qz qw; ARKit reference camera convention')
    loops=read_json(summary['inputs']['loops']['path']);retrieval=read_json(Path(loops['retrieval'])/'report.json')
    spheres={s['anchor_timestamp']:s for s in read_json(summary['inputs']['spheres']['path'])['snapshots']}
    plots(args.output,graph,optimized,edges,summary)
    plot_reference(args.output,reference)
    recording(args.output,graph,optimized,timestamps,spheres,edges,summary,loops,retrieval,reference=reference)
    print(json.dumps(dict(independent_sim3=metrics,shared_optimized=shared_error),indent=2))


if __name__=='__main__':
    main()

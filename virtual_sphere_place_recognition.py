"""Offline source-image SPHORB experiment; no panorama texture enters extraction."""
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from dpvo.virtual_sphere import FAMILY, SavedKeyframes, VirtualSphorbExtractor, sample_texture
from sphere_place_recognition import parser as retrieval_parser, run


def pixel_rays(width, height):
    y,x=np.mgrid[:height,:width]
    longitude=((x+.5)/width-.5)*2*np.pi;latitude=((y+.5)/height-.5)*np.pi
    return np.stack((np.cos(latitude)*np.sin(longitude),np.sin(latitude),np.cos(latitude)*np.cos(longitude)),axis=-1).astype(np.float32)


def main():
    p=retrieval_parser()
    p.description=__doc__
    p.add_argument('--dense-map',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--video',type=Path)
    source.add_argument('--images',type=Path,help='Sorted original images using DPVO image-stream indexing and full-resolution crop')
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--source-cache',type=Path,required=True)
    p.add_argument('--source-stride','--video-stride',dest='video_stride',type=int,default=5)
    p.add_argument('--source-skip','--video-skip',dest='video_skip',type=int,default=0)
    p.add_argument('--mesh-depth-jump',type=float,default=.05)
    p.add_argument('--visibility-depth-band',type=float,default=.03)
    p.add_argument('--snapshot-limit',type=int)
    p.add_argument('--warm-anchors',type=int,nargs='*',default=[336])
    args=p.parse_args()
    if args.output is None or args.downsample!=1 or args.benchmark_passes:
        p.error('provide a fresh --output, use downsample=1 and benchmark-passes=0; selected anchors get three warmed direct-sampling passes')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    args.extractor=FAMILY
    manifest=json.loads((args.spheres/'manifest.json').read_text())
    snapshots=manifest['snapshots'][:args.snapshot_limit]
    if not args.train_spheres<len(snapshots):
        p.error('need query spheres after the vocabulary training window')
    cv2.setNumThreads(args.threads)
    sources=SavedKeyframes(args.dense_map,args.trajectory,args.images if args.images is not None else args.video,args.calibration,args.source_cache,
                          args.video_stride,args.video_skip,args.mesh_depth_jump)
    start=perf_counter();sources.prepare([t for s in snapshots for t in s['source_timestamps']]);source_prepare_seconds=perf_counter()-start
    extractor=VirtualSphorbExtractor(args.features,args.levels,args.threshold,args.threads)
    previews=output/'previews';previews.mkdir()
    (previews/'manifest.json').write_text(json.dumps(dict(manifest,snapshots=snapshots),indent=2)+'\n')
    (output/'features').mkdir();(output/'provenance').mkdir()
    prepared,records,warmed={ },[],[]
    rays=pixel_rays(1024,512).reshape(-1,3)
    for i,snapshot in enumerate(snapshots):
        anchor=snapshot['anchor_timestamp'];start=perf_counter()
        scene,mesh_stats=sources.scene(snapshot,args.visibility_depth_band);scene_seconds=perf_counter()-start
        start=perf_counter();pyramids=[sources.pyramid(t) for t in snapshot['source_timestamps']];source_io_seconds=perf_counter()-start
        start=perf_counter()
        feature,provenance,timing=extractor(scene,pyramids,snapshot['source_timestamps'],
            provenance_dir=output/'grids'/f'{anchor:06d}')
        extraction_seconds=perf_counter()-start-timing['provenance_write_seconds']
        feature.save(output/'features'/f'{anchor:06d}.npz')
        np.savez_compressed(output/'provenance'/f'{anchor:06d}.npz',**provenance)
        timing.update(scene_seconds=scene_seconds,source_io_seconds=source_io_seconds,
                      extraction_seconds=extraction_seconds,**mesh_stats)
        prepared[anchor]=dict(features=feature,extraction_seconds=extraction_seconds,timing=timing)
        records.append(dict(anchor=anchor,features=len(feature.uv),**timing))
        # Diagnostic preview is created only AFTER extraction and never fed back.
        start=perf_counter();hit=scene.sample(rays,args.threads)
        gray,valid,lod=sample_texture(hit,pyramids,2*np.pi/1024)
        image=np.column_stack((gray,gray,gray,valid)).reshape(512,1024,4)
        directory=previews/snapshot['directory'];directory.mkdir()
        if not cv2.imwrite(str(directory/'rgb.png'),image):raise OSError('could not write preview')
        np.savez_compressed(directory/'projection.npz',radial_depth=np.where(valid,hit['depth'],np.nan).reshape(512,1024))
        (directory/'metadata.json').write_text(json.dumps(snapshot,indent=2)+'\n')
        records[-1]['preview_seconds']=perf_counter()-start
        if anchor in args.warm_anchors:
            for pass_index in range(3):
                start=perf_counter();again,other,t=extractor(scene,pyramids,snapshot['source_timestamps'])
                elapsed=perf_counter()-start
                for name in ('descriptors','uv','bearings','points','responses','orientations','sizes','octaves'):
                    np.testing.assert_array_equal(getattr(feature,name),getattr(again,name))
                for name in provenance:np.testing.assert_array_equal(provenance[name],other[name])
                warmed.append(dict(anchor=anchor,pass_index=pass_index,extraction_seconds=elapsed,**t))
        print(f'Virtual sphere {i+1}/{len(snapshots)}, anchor {anchor}: {len(feature.uv)} features, '
              f'{timing["coherent_candidates"]} coherent candidates, {extraction_seconds:.2f}s extraction',flush=True)
        (output/'virtual_input.json').write_text(json.dumps(dict(source=sources.fingerprint,
            source_prepare_seconds=source_prepare_seconds,records=records,warmed=warmed,
            texture_input='original keyframe images only; saved sphere RGB/depth raster is not used',
            visibility='closest surface per source; accept within relative band of global nearest surface',
            source_coherence='extract each visible source separately; suppress duplicate bearings before per-octave quotas',
            provenance='grids contains every valid source sample and its UV/mip/depth/triangle; invalid samples are implicit',
            preview='grayscale visualization generated after extraction; does not enter descriptors'),indent=2)+'\n')
    args.output=output/'retrieval'
    report=run(args,prepared=prepared,visualization_input=previews)
    print('Virtual experiment:',output,flush=True)


if __name__=='__main__':main()

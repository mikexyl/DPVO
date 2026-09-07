import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import cv2
import numpy as np
from dpvo import _sphorb
from dpvo.sphorb import SphorbExtractor
from dpvo.virtual_sphere import SavedKeyframes, depth_mesh, sample_texture


class SavedSourceStreamTests(unittest.TestCase):
    def test_image_and_video_sources_match_actual_dpvo_streams(self):
        from dpvo.stream import image_stream, video_stream
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);images=root/'images';images.mkdir()
            rng=np.random.default_rng(91)
            originals=[rng.integers(0,256,(64,90,3),np.uint8) for _ in range(8)]
            for i,im in enumerate(originals):self.assertTrue(cv2.imwrite(str(images/f'{i:03d}.png'),im))
            video=root/'source.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),30.,(90,64))
            self.assertTrue(writer.isOpened())
            for im in originals:writer.write(im)
            writer.release()
            calibration=root/'calibration.txt';np.savetxt(calibration,[60.,61.,44.,32.])
            for kind,path,stream in [('images',images,image_stream),('video',video,video_stream)]:
                with self.subTest(kind=kind):
                    streamed=[]
                    stream(SimpleNamespace(put=streamed.append),str(path),str(calibration),2,1)
                    streamed=[x for x in streamed if x[0]>=0]
                    timestamps=[0,len(streamed)-1]
                    trajectory=root/f'{kind}.txt'
                    poses=np.zeros((len(streamed),8));poses[:,0]=np.arange(len(streamed));poses[:,-1]=1
                    np.savetxt(trajectory,poses)
                    vertices=[];expected_uv={}
                    xy=np.array([[30,30],[31,30],[30,31],[31,31]],np.int32)
                    for timestamp in timestamps:
                        _,processed,K=streamed[timestamp]
                        grid_uv=(xy*2+.5)*np.array(processed.shape[1::-1])/[504,378]-.5
                        camera=np.column_stack(((grid_uv-K[2:])/K[:2]*2,np.full(4,2.)))
                        rgb=cv2.cvtColor(cv2.resize(processed,(504,378),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                        colors=rgb[xy[:,1]*2,xy[:,0]*2]
                        data=np.empty(4,dtype=[(n,'f4') for n in ('x','y','z')]+[(n,'u1') for n in ('red','green','blue')])
                        for d,n in enumerate(('x','y','z')):data[n]=camera[:,d]
                        for d,n in enumerate(('red','green','blue')):data[n]=colors[:,d]
                        vertices.append(data)
                        expected_uv[timestamp]=grid_uv if kind=='images' else (grid_uv+.5)*2-.5
                    dense=root/f'{kind}.ply'
                    PlyData([PlyElement.describe(np.concatenate(vertices),'vertex')]).write(dense)
                    dense.with_suffix('.json').write_text(json.dumps(dict(point_stride=2,frames=[dict(timestamp=t,points=4) for t in timestamps])))
                    cache=root/f'{kind}-cache'
                    source=SavedKeyframes(dense,trajectory,path,calibration,cache,stride=2,skip=1)
                    source.prepare(timestamps)
                    for timestamp in timestamps:
                        record=json.loads((cache/f'{timestamp:06d}.json').read_text())
                        raw_index=1+2*timestamp if kind=='images' else 1+2*(timestamp+1)-1
                        self.assertEqual(record['raw_frame'],raw_index)
                        self.assertTrue(record['source_color_exact'])
                        np.testing.assert_array_equal(cv2.imread(str(cache/f'{timestamp:06d}.png')),originals[raw_index])
                        np.testing.assert_allclose(source.geometry(timestamp)[2],expected_uv[timestamp],atol=1e-5)
                    # Stride changes cannot silently reuse the old source cache.
                    with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                        SavedKeyframes(dense,trajectory,path,calibration,cache,stride=1,skip=1)


def plane(depths=(2, 2, 2, 2), source=0):
    uv = np.array([[20, 20], [80, 20], [20, 80], [80, 80]], np.float32)
    z = np.array(depths, np.float32)
    v = np.column_stack(((uv - 50) / 100 * z[:, None], z)).astype(np.float32)
    t = np.array([[0, 2, 1], [1, 2, 3]], np.int32)
    return v, t, uv, z, np.full(2, source, np.int32)


class VirtualSphereTests(unittest.TestCase):
    def test_source_projection_under_seam_and_pole_rotations(self):
        import cv2
        v,t,u,z,s=plane((1,2,1,2))
        rays=np.array([[0,0,1],[.1,-.1,1]],np.float32);rays/=np.linalg.norm(rays,axis=1)[:,None]
        expected=_sphorb.VirtualScene(v,t,u,z,s).sample(rays)
        for rotation in (np.array([0,np.pi,0]),np.array([np.pi/2,0,0]),np.array([.3,1.2,.7])):
            R=cv2.Rodrigues(rotation)[0].astype(np.float32)
            actual=_sphorb.VirtualScene(np.ascontiguousarray(v@R.T),t,u,z,s).sample(np.ascontiguousarray(rays@R.T))
            np.testing.assert_allclose(actual['uv'],expected['uv'],atol=3e-5)
            np.testing.assert_allclose(actual['depth'],expected['depth'],rtol=2e-6)
            np.testing.assert_allclose(actual['pixel_scale'],expected['pixel_scale'],rtol=3e-5)
            np.testing.assert_array_equal(actual['visible_sources'],expected['visible_sources'])

    def test_perspective_correct_source_coordinates_and_visibility(self):
        scene = _sphorb.VirtualScene(*plane((1, 2, 1, 2)))
        rays = np.array([[0, 0, 1], [.1, -.1, 1], [0, 0, -1]], np.float32)
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        a, b = scene.sample(rays, 1), scene.sample(rays, 4)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_allclose(a['uv'][:2], [[50, 50], [60, 40]], atol=2e-5)
        self.assertEqual(a['source'][2], -1)
        self.assertTrue(np.isnan(a['depth'][2]))
        layers = [plane(), plane((2.04,)*4, 1), plane((3,)*4, 2)]
        v = np.concatenate([p[0] for p in layers]); t = np.concatenate([p[1]+i*4 for i,p in enumerate(layers)])
        u,z,s = (np.concatenate([p[k] for p in layers]) for k in (2,3,4))
        hit = _sphorb.VirtualScene(v,t,u,z,s).sample(rays[:1])
        self.assertEqual(hit['source'][0], 1)  # Newest within 3%; background stays hidden.
        self.assertAlmostEqual(hit['depth'][0], 2.04, places=5)
        visible_scene=_sphorb.VirtualScene(v,t,u,z,s)
        self.assertEqual(visible_scene.sample(rays[:1],1,0)['source'][0],0)
        self.assertEqual(visible_scene.sample(rays[:1],1,2)['source'][0],-1)
        self.assertEqual(hit['visible_sources'][0],3)
        back=t.copy();back[:2]=back[:2, ::-1]
        blocked=_sphorb.VirtualScene(np.concatenate([layers[0][0],layers[2][0]]),
            np.concatenate([layers[0][1][:, ::-1],layers[2][1]+4]),
            np.concatenate([layers[0][2],layers[2][2]]),np.concatenate([layers[0][3],layers[2][3]]),
            np.concatenate([layers[0][4],layers[2][4]]))
        self.assertEqual(blocked.sample(rays[:1])['source'][0],-1)  # Back surface occludes, but cannot supply texture.
        v[:] = 999  # Constructor owns geometry; output owns arrays.
        np.testing.assert_allclose(scene.sample(rays)['uv'][:2], a['uv'][:2])

    def test_invalid_mesh_rays_and_empty_scene(self):
        v,t,u,z,s = plane()
        bad=t.copy();bad[0,0]=99
        with self.assertRaises(ValueError): _sphorb.VirtualScene(v,bad,u,z,s)
        badz=z.copy();badz[0]=np.nan
        with self.assertRaises(ValueError): _sphorb.VirtualScene(v,t,u,badz,s)
        scene=_sphorb.VirtualScene(v,t,u,z,s)
        with self.assertRaises(ValueError): scene.sample(np.zeros((1,3),np.float32))
        with self.assertRaises(ValueError): scene.sample(np.array([[0,0,1]],np.float32),0)
        empty=_sphorb.VirtualScene(v[:0],t[:0],u[:0],z[:0],s[:0])
        self.assertEqual(empty.sample(np.array([[0,0,1]],np.float32))['source'][0],-1)

    def test_mesh_holes_and_depth_edges(self):
        xy=np.array([[0,0],[1,0],[0,1],[1,1]],np.int32)
        points=np.column_stack((xy,np.ones(4))).astype(np.float32)
        self.assertEqual(len(depth_mesh(points,xy,(2,2))),2)
        self.assertEqual(len(depth_mesh(points[:3],xy[:3],(2,2))),0)
        points[3,2]=2
        self.assertEqual(len(depth_mesh(points,xy,(2,2))),0)

    def test_texture_filtering_and_valid_black(self):
        image=np.zeros((100,100),np.uint8)
        hit=dict(source=np.array([0,0,-1]), uv=np.array([[50,50],[-1,20],[0,0]],np.float32),pixel_scale=np.ones(3))
        gray,valid,lod=sample_texture(hit,[[image]],1)
        np.testing.assert_array_equal(gray,[0,0,0]);np.testing.assert_array_equal(valid,[255,0,0])

    def test_direct_grid_source_coherence_ownership_threads(self):
        a=SphorbExtractor(levels=2,workers=1).native;b=SphorbExtractor(levels=2,workers=4).native
        rng=np.random.default_rng(16)
        grids=[rng.integers(0,256,a.grid_bearings(l).shape[:-1],np.uint8) for l in range(2)]
        masks=[np.full(g.shape,255,np.uint8) for g in grids]
        sources=[np.ones(g.shape,np.uint8) for g in grids]
        first=a.extract_grids(grids,masks,sources); second=b.extract_grids(grids,masks,sources)
        self.assertGreater(len(first['uv']),100)
        for k in first:
            if k!='native_seconds': np.testing.assert_array_equal(first[k],second[k])
        saved=first['descriptors'].copy()
        for mask in masks: mask[:, :50, :80]=0
        masked=a.extract_grids(grids,masks,sources)
        for g,mask in zip(grids,masks):g[mask==0]=255-g[mask==0]
        hidden=b.extract_grids(grids,masks,sources)
        np.testing.assert_array_equal(masked['descriptors'],hidden['descriptors'])
        np.testing.assert_array_equal(masked['uv'],hidden['uv'])
        for s in sources: s[...,::2]=2  # Every feature support crosses source boundaries.
        self.assertEqual(len(a.extract_grids(grids,masks,sources)['uv']),0)
        for mask in masks: mask[:]=0
        self.assertEqual(len(b.extract_grids(grids,masks,sources)['uv']),0)
        np.testing.assert_array_equal(first['descriptors'],saved)
        with self.assertRaises(ValueError): a.extract_grids([grids[0]],masks,sources)


if __name__=='__main__': unittest.main()

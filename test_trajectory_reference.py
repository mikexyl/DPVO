from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dpvo.trajectory_reference import align_positions,apply_alignment,position_errors,video_reference


class TrajectoryReferenceTests(unittest.TestCase):
    def write_csv(self,path,ids):
        rows=[f'{10+frame/30},{frame},{frame},0,1,0,0,0,1' for frame in ids]
        path.write_text('timestamp,frame,x,y,z,qx,qy,qz,qw\n'+'\n'.join(rows)+'\n')

    def test_exact_video_frame_mapping_with_skip_and_reordered_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.csv'
            self.write_csv(path,[41,6,16,999])
            ref=video_reference(path,np.array([0,2,7]),stride=5,skip=2)
            np.testing.assert_array_equal(ref['raw_frame_ids'],[6,16,41])
            np.testing.assert_array_equal(ref['csv_rows'],[1,2,0])
            np.testing.assert_array_equal(ref['positions'][:,0],[6,16,41])
            self.assertEqual(ref['positions'].shape,(3,3))

    def test_missing_duplicate_and_noninteger_frames_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.csv'
            for ids in ([0,1,2],[0,1,1],[0,1,2.5]):
                self.write_csv(path,ids)
                with self.assertRaises(ValueError):video_reference(path,np.array([0,1,2]),stride=5,skip=0)

    def test_global_sim3_and_metric_rmse(self):
        rng=np.random.default_rng(27);x=rng.normal(size=(20,3))
        rotation=Rotation.from_euler('xyz',[15,80,-25],degrees=True).as_matrix()
        y=3.4*x@rotation.T+[2,-1,8]
        aligned,fit=align_positions(x,y)
        np.testing.assert_allclose(aligned,y,atol=1e-12)
        np.testing.assert_allclose(fit['rotation'],rotation,atol=1e-12)
        self.assertAlmostEqual(fit['scale'],3.4)
        error,stats=position_errors(aligned,y)
        self.assertLess(stats['rmse_m'],1e-12)
        shifted=apply_alignment(x+np.array([1.,0,0]),fit)
        _,stats=position_errors(shifted,y)
        self.assertAlmostEqual(stats['rmse_m'],3.4)


if __name__=='__main__':unittest.main()

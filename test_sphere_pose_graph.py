import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from dpvo.sphere_pose_graph import (Graph, as_sim3, edge_errors, from_tum,
                                    measurement, optimize_graph, robust_residual)


def chain(poses, loop_measurements=(), loop_targets=(), confidence=None):
    n = len(poses)
    source, target = list(range(1,n)), list(range(n-1))
    states = as_sim3(poses)
    z = list((states[target].inv()*states[source]).data.numpy())
    sigma = [np.full(7,.05) for _ in source]
    for measurement_value, j in zip(loop_measurements,loop_targets):
        source.append(n-1);target.append(j);z.append(measurement_value);sigma.append(np.full(7,.03))
    flags = np.arange(len(source)) >= n-1
    weights = np.ones(len(source))
    if confidence is not None: weights[flags] = confidence
    return Graph(poses, np.array(source), np.array(target), np.array(z), np.array(sigma), flags, weights)


class SpherePoseGraphTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(43)

    def random_poses(self,n):
        values = self.rng.normal(size=(n,7))*.3
        from dpvo.lietorch import Sim3
        return Sim3.exp(torch.from_numpy(values)).data.numpy()

    def test_direction_scale_opposite_rotation_and_world_gauge(self):
        # Independent matrices/point actions check the exact edge convention.
        values = self.random_poses(2)
        values[0] = measurement(Rotation.from_euler('y',179.5,degrees=True).as_matrix(),[2,-1,3],2.4)
        values[1] = measurement(np.eye(3),[-1,.4,2],.7)
        source = self.rng.normal(size=(20,3));source[:,2] -= 2  # Rear hemisphere allowed.
        R = Rotation.from_quat(values[:,3:7]).as_matrix()
        world = values[0,7]*(source@R[0].T)+values[0,:3]
        target = (world-values[1,:3])@R[1]/values[1,7]
        rz = R[1].T@R[0]; sz = values[0,7]/values[1,7]
        tz = R[1].T@(values[0,:3]-values[1,:3])/values[1,7]
        np.testing.assert_allclose(sz*(source@rz.T)+tz,target,atol=1e-12)
        graph = Graph(values,np.array([0]),np.array([1]),measurement(rz,tz,sz)[None],
                      np.ones((1,7)),np.array([True]),np.ones(1))
        np.testing.assert_allclose(edge_errors(graph,values),0,atol=1e-12)
        common = as_sim3(self.random_poses(1))
        transformed = (common*as_sim3(values)).data.numpy()
        np.testing.assert_allclose(edge_errors(graph,transformed),0,atol=1e-12)
        wrong = graph.measurements.copy()
        graph.measurements = as_sim3(wrong).inv().data.numpy()
        self.assertGreater(np.linalg.norm(edge_errors(graph,values)),1)

    def test_no_loop_preserves_input_gauge_and_ownership(self):
        poses = self.random_poses(5); before = poses.copy(); graph = chain(poses)
        threads = torch.get_num_threads()
        result, summary = optimize_graph(graph, workers=1)
        self.assertTrue(summary['success'])
        np.testing.assert_allclose(result,poses,atol=1e-12)
        np.testing.assert_array_equal(graph.initial[0],result[0])
        np.testing.assert_array_equal(poses,before)
        self.assertEqual(torch.get_num_threads(),threads)
        result[1,0] += 1
        np.testing.assert_array_equal(graph.initial,poses)

    def test_loop_recovers_synthetic_scale_drift(self):
        # A chain with a biased final position/scale, constrained by an exact loop.
        n=8; poses=np.zeros((n,8)); poses[:,6:]=1
        poses[:,0]=np.linspace(0,1.4,n);poses[:,7]=np.exp(np.linspace(0,.35,n))
        truth = poses.copy(); truth[:,0]=np.linspace(0,1,n);truth[:,7]=1
        z=(as_sim3(truth)[0].inv()*as_sim3(truth)[-1]).data.numpy()
        graph=chain(poses,[z],[0]);err0=np.linalg.norm(edge_errors(graph,poses)[-1])
        result,summary=optimize_graph(graph,workers=1,max_nfev=150)
        self.assertTrue(summary['success'],summary)
        self.assertLess(np.linalg.norm(edge_errors(graph,result)[-1]),err0*.1)
        self.assertLess(abs(result[-1,7]-1),abs(poses[-1,7]-1)*.2)
        self.assertLess(abs(result[-1,0]-1),abs(poses[-1,0]-1)*.2)
        self.assertLess(summary['final_cost'],summary['initial_cost'])
        again,_=optimize_graph(graph,workers=2,max_nfev=150)
        np.testing.assert_allclose(result,again,rtol=1e-7,atol=1e-8)

    def test_block_robust_loss_limits_bad_loop(self):
        poses=np.zeros((5,8));poses[:,6:]=1;poses[:,0]=np.arange(5)*.2
        good=measurement(np.eye(3),[.8,0,0],1)
        bad=measurement(Rotation.from_euler('y',150,degrees=True).as_matrix(),[20,5,3],3)
        graph=chain(poses,[good,bad],[0,1],confidence=[1,.05])
        robust,_=optimize_graph(graph,delta=3,workers=1,max_nfev=150)
        quadratic,_=optimize_graph(graph,delta=1e8,workers=1,max_nfev=150)
        self.assertLess(np.linalg.norm(robust[:,:3]-poses[:,:3]),
                        np.linalg.norm(quadratic[:,:3]-poses[:,:3])*.15)
        self.assertLess(np.linalg.norm(robust[-1,:3]-poses[-1,:3]),.03)
        errors=edge_errors(graph,poses);r=robust_residual(errors,graph,3)
        z=errors/graph.sigmas;expected=.5*np.sum(z[~graph.loops]**2)
        expected+=np.sum(graph.confidence[graph.loops]*9*(np.sqrt(1+np.sum(z[graph.loops]**2,axis=1)/9)-1))
        self.assertAlmostEqual(.5*np.sum(r*r),expected,places=8)

    def test_reject_corrupt_disconnected_inputs(self):
        poses=self.random_poses(3)
        with self.assertRaisesRegex(ValueError,'connected'):
            Graph(poses,np.array([0]),np.array([1]),poses[:1],np.ones((1,7)),np.ones(1,bool),np.ones(1))
        bad=poses.copy();bad[0,7]=-1
        with self.assertRaises(ValueError): chain(bad)
        with self.assertRaises(ValueError): measurement(np.diag([1,1,-1]),[0,0,0],1)
        with self.assertRaises(ValueError): from_tum(np.zeros((2,8)))


if __name__=='__main__':
    unittest.main()

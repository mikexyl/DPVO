"""Accepted TEASER rotations must work with older SciPy on JetPack 6."""
from collections import Counter
from types import SimpleNamespace as NS
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dpvo.loop_closure.distributed import DistributedLongTermLoopClosure, FrameIdentity


class ReadonlyRotationTest(unittest.TestCase):
    def test_accepted_readonly_rotation_publishes_constraint(self):
        rotation = np.eye(3)
        rotation.flags.writeable = False
        payload = NS(frame=FrameIdentity('robot1', 'session', 0), pose=np.zeros(7),
                     points=np.array([[0., 0., 1.]] * 30),
                     keypoints=np.zeros((30, 2)), descriptors=np.zeros((30, 64)),
                     image_size=np.array([384, 240]))
        matches = torch.arange(30).repeat(2, 1).T
        backend = NS(diagnostics=Counter(), _local_payload=lambda _: payload,
                     cfg=NS(MULTI_ROBOT_MAX_DEPTH=20, MULTI_ROBOT_MIN_INLIERS=30),
                     matcher=lambda _: {'matches': [matches]},
                     verifier=NS(verify=lambda *_: NS(success=True, rotation=rotation,
                         translation=np.zeros(3), scale=1., inliers=30,
                         inlier_ratio=1., method='teaser++')),
                     transport=Mock(), distributed_lock=threading.RLock(),
                     candidate_detector=Mock(), inter_robot_lc_count=0,
                     _record_verification=Mock(), _diagnostic=Mock())
        match = NS(local_keyframe_id=0, score=.8)
        def legacy_from_matrix(matrix):
            if not matrix.flags.writeable:
                raise ValueError('buffer source array is read-only')
            return Rotation.from_matrix(matrix)
        with patch.object(torch.Tensor, 'cuda', lambda self: self), patch(
                'dpvo.loop_closure.distributed.Rotation') as legacy:
            legacy.from_matrix.side_effect = legacy_from_matrix
            self.assertTrue(DistributedLongTermLoopClosure._verify_remote_match(
                backend, match, payload))
        constraint = backend.transport.publish_constraint.call_args.args[0]
        np.testing.assert_allclose(constraint.quaternion_xyzw, [0, 0, 0, 1])
        self.assertEqual(backend.diagnostics['accepted'], 1)
        self.assertFalse(rotation.flags.writeable)

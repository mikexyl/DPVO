"""CPU tests for distributed epoch/round isolation (ROS runtime required)."""
import json
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
try:
    from dpvo_multi_robot.cbs_agent import CbsAgent
    from dpvo_multi_robot.distributed_cbs import DistributedCbsNode
    ROS = True
except ImportError:
    ROS = False


@unittest.skipUnless(ROS, 'Requires ROS overlay')
class CbsProtocolTest(unittest.TestCase):
    def agent(self):
        n = object.__new__(CbsAgent)
        n.robot, n.epoch = 'robot0', 'epoch2'
        n.job = dict(sessions={'robot0':'s0', 'robot1':'s1'}, iterations=40)
        n.lock = threading.RLock()
        n.inbox, n.outgoing = {}, {}
        n.belief_pub = Mock()
        n.cancel = threading.Event()
        n.released = False
        return n

    def test_rejects_old_epoch_unknown_sender_and_invalid_round(self):
        n = self.agent()
        base = dict(epoch='epoch2', sender='robot1', round=0, beliefs=[])
        for change in ({'epoch':'epoch1'}, {'sender':'robot2'}, {'round':-1}, {'round':40}):
            n.on_belief(NS(data=json.dumps(dict(base, **change))))
        self.assertEqual(n.inbox, {})
        n.on_belief(NS(data=json.dumps(base)))
        n.on_belief(NS(data=json.dumps(dict(base, beliefs=[1]))))
        self.assertEqual(n.inbox[0]['robot1']['beliefs'], [])

    def test_retransmits_frozen_snapshot_without_advancing_optimizer(self):
        n = self.agent()
        n.outgoing[3] = dict(epoch='epoch2', sender='robot0', round=3, beliefs=[])
        n.on_belief(NS(data=json.dumps(dict(epoch='epoch2', sender='robot1', round=3, need='robot0'))))
        sent = json.loads(n.belief_pub.publish.call_args.args[0].data)
        self.assertEqual(sent, n.outgoing[3])
        self.assertEqual(n.inbox, {})

    def test_release_and_cancel_are_epoch_scoped(self):
        n = self.agent()
        n.on_release(NS(data=json.dumps(dict(epoch='old', cancel=True))))
        self.assertFalse(n.cancel.is_set())
        n.on_release(NS(data=json.dumps(dict(epoch='epoch2'))))
        self.assertTrue(n.released)
        n.on_release(NS(data=json.dumps(dict(epoch='epoch2', cancel=True))))
        self.assertTrue(n.cancel.is_set())


@unittest.skipUnless(ROS, 'Requires ROS overlay')
class CbsCommitTest(unittest.TestCase):
    def test_previews_are_ignored_until_final_results(self):
        node = object.__new__(DistributedCbsNode)
        node.lock = threading.RLock()
        node.robot_ids = ['robot0', 'robot1']
        node.current_epoch = 'epoch'
        node.results, node.alignments = {}, {}
        node._report_alignment = Mock()
        for iteration in range(1, 101):
            for robot in node.robot_ids:
                node.agent_result(NS(data=json.dumps(dict(epoch='epoch', robot=robot,
                    iteration=iteration, provisional=True))))
        self.assertEqual(node.results, {})
        self.assertEqual(node.alignments, {})
        node._report_alignment.assert_not_called()
        node.agent_result(NS(data=json.dumps(dict(epoch='old', robot='robot0'))))
        self.assertEqual(node.results, {})
        node.agent_result(NS(data=json.dumps(dict(epoch='epoch', robot='robot0'))))
        self.assertEqual(set(node.results), {'robot0'})
        node._report_alignment.assert_not_called()

    def test_restart_between_wait_and_commit_rejects_old_results(self):
        node = object.__new__(DistributedCbsNode)
        node.dispatch_closed = False
        node.active_sessions = {'robot0': 'new'}
        node.results = {'robot0': dict(sessions={'robot0': 'old'},
            optimizer_instances=1, translation=[0, 0, 0],
            quaternion=[0, 0, 0, 1], scale=1)}
        with self.assertRaisesRegex(RuntimeError, 'restarted'):
            node._validate_results({'robot0': 'old'})
        node.active_sessions['robot0'] = 'old'
        node._validate_results({'robot0': 'old'})

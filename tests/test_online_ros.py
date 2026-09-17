"""Session and snapshot regressions; run with the ROS overlay sourced."""
from pathlib import Path
import sys
import threading
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros2/dpvo_multi_robot'))
try:
    from dpvo_multi_robot.online_common import fleet_frame
    from dpvo_multi_robot.online_cbs import OnlineCbsNode
    from dpvo_multi_robot.cbs_pgo import CbsPgoNode
    from dpvo_multi_robot.online_viewer import FleetViewer
    from dpvo_multi_robot.online_control import OnlineControl
    from dpvo_multi_robot_interfaces.msg import RobotMapTransform
    from nav_msgs.msg import Path as RosPath
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import PointCloud2, CompressedImage
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, 'Source ROS and build the interfaces; install Viser')
class OnlineSessionTest(unittest.TestCase):
    def cbs(self):
        node = object.__new__(OnlineCbsNode)
        node.lock = threading.RLock()
        node.robot_ids = ['r0', 'r1']
        node.paths = {}
        node.constraints = []
        node.constraint_ids = set()
        node.active_sessions = {}
        node.sessions_by_robot = {}
        node.done_robots = set()
        node.online_transforms = {}
        node.revision = 0
        node.worker = None
        node.last_signature = None
        node.get_logger = Mock(return_value=Mock())
        node.get_parameter = Mock(return_value=NS(value=0.))
        return node

    def test_restart_clears_old_constraints_and_rejects_old_path(self):
        node = self.cbs()
        for robot in node.robot_ids:
            node._activate_session(robot, '001')
            message = RosPath()
            message.header.frame_id = f'{robot}/001/map'
            message.poses = [PoseStamped()]
            node._path(robot, message)
        node.constraints = [NS(query_robot=('r0', '001'), match_robot=('r1', '001'))]
        self.assertTrue(node._ready()[0])  # No player-done messages needed.
        self.assertTrue(node._activate_session('r0', '002'))
        self.assertFalse(node._ready()[0])
        self.assertEqual(node.constraints, [])
        message.header.frame_id = 'r0/001/map'
        node._path('r0', message)
        self.assertNotIn('r0', node.paths)

    def test_partial_fleet_and_missing_anchor_can_optimize(self):
        node = self.cbs()
        node.robot_ids = ['r0', 'r1', 'r2']
        for robot in ('r1', 'r2'):
            message = RosPath()
            message.header.frame_id = f'{robot}/001/map'
            message.poses = [PoseStamped()]
            node._path(robot, message)
        node.constraints = [NS(query_robot=('r1', '001'), match_robot=('r2', '001'))]
        self.assertTrue(node._ready()[0])
        members, _ = node._eligible(node.constraints, {'r1': [1], 'r2': [1]}, node.active_sessions)
        self.assertEqual(members, {'r1', 'r2'})
        node.constraints[0].match_robot = ('r2', '000')
        self.assertFalse(node._ready()[0])

    def test_snapshot_filters_isolated_robot_before_solver(self):
        node = self.cbs()
        constraints = [NS(query_robot=('r0', '001'), match_robot=('r1', '001'))]
        with patch.object(CbsPgoNode, '_run') as run:
            node._run(constraints, {'r0': [1], 'r1': [1], 'r2': [1]},
                      {'r0': '001', 'r1': '001', 'r2': '001'}, {},
                      {'r0': [0.], 'r1': [0.], 'r2': [0.]})
            self.assertEqual(set(run.call_args.args[1]), {'r0', 'r1'})
            self.assertEqual(set(run.call_args.args[2]), {'r0', 'r1'})
            self.assertEqual(set(run.call_args.args[4]), {'r0', 'r1'})

    def test_snapshot_timestamps_are_frozen_at_launch(self):
        node = self.cbs()
        for robot in node.robot_ids:
            message = RosPath()
            message.header.frame_id = f'{robot}/001/map'
            pose = PoseStamped()
            pose.pose.orientation.w = 1.
            pose.header.stamp.sec = 12
            message.poses = [pose]
            node._path(robot, message)
        node.constraints = [NS(query_robot=('r0', '001'), match_robot=('r1', '001'))]
        with patch('dpvo_multi_robot.cbs_pgo.threading.Thread') as thread:
            self.assertTrue(node._launch(force=True)[0])
            snapshot = thread.call_args.kwargs['args']
            node.paths['r0'].poses[0].header.stamp.sec = 99
            self.assertEqual(snapshot[4]['r0'], [12.])

    def test_stale_solver_result_is_not_published(self):
        node = self.cbs()
        node.active_sessions = {'r0': '002'}
        graph = NS(vertices=[NS(robot_id='r0', session_id='001')])
        with patch.object(CbsPgoNode, '_publish_results') as publish:
            node._publish_results(graph, None, None, 1.)
            publish.assert_not_called()

    def viewer(self):
        node = object.__new__(FleetViewer)
        node.lock = threading.RLock()
        node.states = {'r0': dict(session=None, transform=None, frame=NS(), label=NS(),
            cloud=NS(), dense=NS(), raw_dense=np.empty((0, 3), dtype=np.float32), path=NS(), offset=np.array([0., 0., 0.]))}
        return node

    def test_preview_rejects_stale_sessions_out_of_order_and_invalid_jpeg(self):
        import io
        from PIL import Image
        node = self.viewer()
        node.health, node.preview_seen = {}, {}
        node.previews = {'r0': NS(image=None)}
        stream = io.BytesIO()
        Image.fromarray(np.full((12, 16, 3), 120, np.uint8)).save(stream, format='JPEG')
        message = CompressedImage(format='jpeg', data=stream.getvalue())
        message.header.frame_id = 'r0/002/map'
        message.header.stamp.sec = 3
        node.preview('r0', message)
        self.assertEqual(node.previews['r0'].image.shape, (12, 16, 3))
        received = node.preview_seen['r0']
        message.header.stamp.sec = 2
        node.preview('r0', message)
        self.assertEqual(node.preview_seen['r0'], received)
        message.header.frame_id = 'r0/001/map'
        node.preview('r0', message)
        self.assertEqual(node.states['r0']['session'], '002')
        message.header.frame_id = 'r0/003/map'
        message.data = b'bad jpeg'
        node.preview('r0', message)
        self.assertEqual(node.states['r0']['session'], '002')

    def test_controls_follow_shared_status_and_disable_when_disconnected(self):
        node = object.__new__(FleetViewer)
        node.health = {'r0': {'state': 'stopped'}}
        node.last_seen = {'r0': 10.}
        node.control_pending = {}
        node.control_buttons = {('r0', action): NS(disabled=None) for action in ('start', 'stop')}
        node.dense_controls = {'r0': NS(disabled=None)}
        node.refresh_controls('r0', 11.)
        self.assertFalse(node.control_buttons['r0', 'start'].disabled)
        self.assertTrue(node.control_buttons['r0', 'stop'].disabled)
        node.health['r0']['state'] = 'worker_running'
        node.refresh_controls('r0', 11.)
        self.assertTrue(node.control_buttons['r0', 'start'].disabled)
        self.assertFalse(node.control_buttons['r0', 'stop'].disabled)
        self.assertTrue(node.dense_controls['r0'].disabled)
        node.refresh_controls('r0', 14.)
        self.assertTrue(node.control_buttons['r0', 'start'].disabled)
        self.assertTrue(node.control_buttons['r0', 'stop'].disabled)

    def test_control_timeout_allows_reconnect(self):
        node = object.__new__(FleetViewer)
        node.lock = threading.RLock()
        node.commands, node.last_seen = [], {}
        node.statuses = {'r0': NS(content='')}
        old = Mock()
        node.control_pending = {'r0': (old, 0.)}
        node.tick()
        old.cancel.assert_called_once()
        self.assertNotIn('r0', node.control_pending)

    def test_dense_control_requires_idle_robot_and_prepared_engine(self):
        node = object.__new__(OnlineControl)
        node.worker = NS(running=False)
        node.report = Mock()
        node.dense_enabled = False
        with tempfile.TemporaryDirectory() as directory:
            node.dense_engine = Path(directory) / 'da3.engine'
            response = node.set_dense(NS(data=True), NS())
            self.assertFalse(response.success)
            self.assertFalse(node.dense_enabled)
            node.dense_engine.write_bytes(b'engine')
            node.dense_engine.with_name('manifest.json').write_text('{}')
            self.assertTrue(node.set_dense(NS(data=True), NS()).success)
            node.worker.running = True
            self.assertFalse(node.set_dense(NS(data=False), NS()).success)
            self.assertTrue(node.dense_enabled)
            node.worker.running = False
            self.assertTrue(node.set_dense(NS(data=False), NS()).success)
            self.assertFalse(node.dense_enabled)

    def test_dense_cloud_shares_alignment_and_clears_on_restart(self):
        node = self.viewer()
        packed = np.array([(1., 2., 3., 0x123456)],
                          dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
        cloud = PointCloud2(height=1, width=1, point_step=16, row_step=16, data=packed.tobytes())
        cloud.header.frame_id = 'r0/001/map'
        node.points('r0', cloud, dense=True)
        state = node.states['r0']
        self.assertTrue(state['dense'].visible)
        np.testing.assert_array_equal(state['dense'].colors, [[18, 52, 86]])
        msg = RobotMapTransform(robot_id='r0', session_id='001', scale=2.)
        msg.header.frame_id = fleet_frame({'r0': '001'})
        msg.local_map_to_global.rotation.w = 1.
        node.transform(msg)
        np.testing.assert_allclose(state['dense'].points, [[2., 4., 6.]])
        node.activate('r0', '002')
        self.assertFalse(state['dense'].visible)
        self.assertEqual(len(state['raw_dense']), 0)
        node.points('r0', cloud, dense=True)
        self.assertFalse(state['dense'].visible)

    def test_transform_scales_points_and_restart_discards_alignment(self):
        node = self.viewer()
        self.assertTrue(node.activate('r0', '001'))
        state = node.states['r0']
        state['raw_points'] = np.array([[1., 2., 3.]])
        msg = RobotMapTransform(robot_id='r0', session_id='001', scale=2.)
        msg.header.frame_id = fleet_frame({'r0': '001'})
        msg.local_map_to_global.rotation.w = 1.
        msg.local_map_to_global.translation.x = 4.
        node.transform(msg)
        np.testing.assert_array_equal(state['cloud'].points, [[2., 4., 6.]])
        self.assertEqual(state['frame'].position, (4., 0., 0.))
        self.assertTrue(node.activate('r0', '002'))
        node.transform(msg)
        self.assertEqual(state['scale'], 1.)
        self.assertFalse(state['cloud'].visible)
        self.assertFalse(node.activate('r0', '001'))
        msg.session_id = '002'
        msg.scale = float('nan')
        node.transform(msg)
        np.testing.assert_array_equal(state['frame'].position, [0., 0., 0.])

    def test_partial_alignment_status_and_peer_restart(self):
        import json
        node = self.viewer()
        node.alignment_labels = {'r0': NS(content='')}
        node.alignment_sessions, node.alignment_groups = {}, {}
        node.activate('r0', '001')
        node.states['r0']['raw_points'] = np.array([[1., 2., 3.]])
        data = dict(sessions={'r0': '001', 'r1': '001', 'r2': '005'},
            alignments={'r0': dict(sessions={'r0': '001', 'r1': '001'},
                translation=[4., 0., 0.], quaternion=[0., 0., 0., 1.], scale=2.)})
        node.alignment_status(NS(data=json.dumps(data)))
        self.assertIn('Merged with r1', node.alignment_labels['r0'].content)
        self.assertEqual(node.states['r0']['scale'], 2.)
        data['alignments']['r0'].update(provisional=True, iteration=7, iterations=100)
        node.alignment_status(NS(data=json.dumps(data)))
        self.assertIn('7/100', node.alignment_labels['r0'].content)
        self.assertIn('7/100', node.states['r0']['label'].text)
        data['alignments']['r0']['provisional'] = False
        node.alignment_status(NS(data=json.dumps(data)))
        self.assertIn('Merged with r1', node.alignment_labels['r0'].content)
        data['sessions']['r2'] = '006'
        node.alignment_status(NS(data=json.dumps(data)))
        self.assertEqual(node.states['r0']['scale'], 2.)
        data['sessions']['r1'] = '002'
        node.alignment_status(NS(data=json.dumps(data)))
        self.assertEqual(node.states['r0']['scale'], 1.)
        self.assertIn('Unmerged', node.alignment_labels['r0'].content)
        data['sessions']['r1'] = '001'  # Delayed old state cannot resurrect alignment.
        node.alignment_status(NS(data=json.dumps(data)))
        self.assertEqual(node.states['r0']['scale'], 1.)

    def test_malformed_cloud_does_not_crash_or_activate_session(self):
        node = self.viewer()
        msg = PointCloud2()
        msg.header.frame_id = 'r0/001/map'
        msg.point_step, msg.width, msg.height, msg.row_step = 16, 1, 1, 16
        msg.data = b'incomplete'
        node.points('r0', msg)
        self.assertIsNone(node.states['r0']['session'])


if __name__ == '__main__':
    unittest.main()

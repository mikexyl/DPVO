"""CPU-only integration check; no camera, CUDA, or Jetson connection.

Run in the coordinator image with ROS_DOMAIN_ID=173 and
ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST; mount the fleet config at /fleet.
The test creates temporary synthetic worker processes and shuts them down.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.request import urlopen

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from std_srvs.srv import Trigger

from dpvo_multi_robot.online_cbs import OnlineCbsNode
from dpvo_multi_robot.online_control import OnlineControl
from dpvo_multi_robot.online_viewer import FleetViewer
from dpvo_multi_robot_interfaces.msg import InterRobotLoopClosure


def main():
    with tempfile.TemporaryDirectory(prefix='dpvo-fleet-smoke-') as directory:
        parameter_file = Path(directory) / 'worker.yaml'
        parameter_file.write_text('{}')
        rclpy.init(args=['--ros-args', '-r', '__ns:=/robot0',
            '-p', f'worker_parameters:={parameter_file}', '-p', 'web_port:=19091',
            '-p', f'output_dir:={directory}/cbs', '-p', 'cbs_executable:=/opt/cbs/run',
            '-p', 'online_period:=1.0', '-p', 'iterations:=5',
            '-p', 'robot_ids:=[robot0, robot1, robot2]',
            '-p', 'run_centralized_baseline:=false',
            '-p', 'run_explicit_anchor_centralized_baseline:=false'])
        executor = SingleThreadedExecutor()
        control = viewer = local = cbs = None
        def until(predicate, timeout=10):
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() > deadline:
                    raise AssertionError('Timed out waiting for ROS state')
                executor.spin_once(timeout_sec=.05)
        try:
            control = OnlineControl()
            # Exercise exactly the real process manager, using a CPU sleeper.
            control.worker.command = [sys.executable, '-c', 'import time; time.sleep(60)']
            viewer = FleetViewer()
            local = FleetViewer(use_global_arguments=False, parameter_overrides=[
                Parameter('robot_ids', value=['robot0']), Parameter('web_port', value=19092)])
            # A width of 2 in world units hides the map behind giant path blobs.
            assert all(state['path'].thickness_units == 'screen'
                       for state in viewer.states.values())
            cbs = OnlineCbsNode()
            for node in (control, viewer, local, cbs):
                executor.add_node(node)
            with urlopen('http://127.0.0.1:19091', timeout=10) as response:
                assert response.status == 200
            assert not control.worker.running
            pids = []
            for cycle in range(2):
                for action in ('start', 'stop'):
                    panel = (local if cycle == 0 else viewer) if action == 'start' else (viewer if cycle == 0 else local)
                    client = panel.control_clients['robot0', action]
                    until(client.service_is_ready)
                    result = client.call_async(Trigger.Request())
                    until(result.done)
                    assert result.result().success, result.result().message
                    expected = 'worker_running' if action == 'start' else 'stopped'
                    until(lambda: all(p.health.get('robot0', {}).get('state') == expected for p in (viewer, local)))
                    for p in (viewer, local):
                        p.refresh_controls('robot0')
                        assert p.control_buttons['robot0', 'start'].disabled == (action == 'start')
                        assert p.control_buttons['robot0', 'stop'].disabled == (action == 'stop')
                    if action == 'start':
                        assert control.worker.running
                        pids.append(control.worker.process.pid)
                        repeated = viewer.control_clients['robot0', 'start'].call_async(Trigger.Request())
                        until(repeated.done)
                        assert repeated.result().success
                        assert control.worker.process.pid == pids[-1]
                    else:
                        assert not control.worker.running
                        try:
                            os.kill(pids[-1], 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise AssertionError('Stopped worker survived')
            assert pids[0] != pids[1]
            assert 'torch' not in sys.modules and 'pyrealsense2' not in sys.modules
            publishers = []
            for robot in ('robot0', 'robot1'):
                publisher = control.create_publisher(RosPath, f'/{robot}/dpvo/path', 1)
                publishers.append(publisher)
                until(lambda: publisher.get_subscription_count() >= 2)
                path = RosPath()
                path.header.frame_id = f'{robot}/00001/map'
                for index in range(3):
                    pose = PoseStamped()
                    pose.header.stamp.sec = index + 1
                    pose.pose.position.x = float(index)
                    pose.pose.orientation.w = 1.
                    path.poses.append(pose)
                publisher.publish(path)
            until(lambda: len(cbs.paths) == 2 and
                  all(viewer.states[r]['session'] == '00001' for r in ('robot0', 'robot1')))
            loops = control.create_publisher(InterRobotLoopClosure, '/dpvo_multi_robot/loop_closure', 10)
            until(lambda: loops.get_subscription_count() >= 1)
            loop = InterRobotLoopClosure(query_robot_id='robot0', match_robot_id='robot1',
                query_session_id='00001', match_session_id='00001',
                query_pose=[0., 0., 0., 0., 0., 0., 1.],
                match_pose=[0., 0., 0., 0., 0., 0., 1.], scale=1.,
                inliers=50, inlier_ratio=1., verification_method='synthetic-smoke')
            loop.query_to_match.rotation.w = 1.
            loops.publish(loop)
            until(lambda: all(viewer.states[r]['label'].text.endswith('CBS aligned')
                              for r in ('robot0', 'robot1')), timeout=45)
            assert not viewer.states['robot2'].get('merged_sessions')
            until(lambda: 'Merged with robot1' in local.alignment_labels['robot0'].content)
            print(json.dumps({'partial_fleet_with_robot2_absent': 'ok', 'http': 'ok', 'fresh_worker_pids': pids,
                'cross_panel_start_stop': 'ok', 'shared_state_and_single_worker': 'ok',
                'idle_gpu_imports': False, 'ros_paths_and_loop': 'ok',
                'cbs_solver_and_viewer_alignment': 'ok'}), flush=True)
        finally:
            if control is not None:
                control.worker.stop()
            if cbs is not None:
                cbs.close()
            if viewer is not None:
                viewer.server.stop()
            if local is not None:
                local.server.stop()
            executor.shutdown()
            for node in (control, viewer, local, cbs):
                if node is not None:
                    node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()

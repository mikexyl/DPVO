"""Real ROS exchange test with native agents; run on an isolated ROS domain."""
import json
from pathlib import Path
import sys
import tempfile
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped
from dpvo_multi_robot.distributed_cbs import DistributedCbsNode
from dpvo_multi_robot.cbs_agent import CbsAgent
from dpvo_multi_robot_interfaces.msg import InterRobotLoopClosure


def main():
    # --remote uses separately launched agent containers on the Jetsons.
    remote = '--remote' in sys.argv
    count = int(sys.argv[sys.argv.index('--count')+1]) if '--count' in sys.argv else 3
    robots = [f'robot{i}' for i in range(count)]
    agents = []
    with tempfile.TemporaryDirectory() as directory:
        from rclpy.node import Node
        rclpy.init(args=['--ros-args', '-p', 'robot_ids:=[robot0,robot1,robot2]',
            '-p', f'output_dir:={directory}', '-p', 'iterations:=40', '-p', 'online_period:=300.0',
            '-p', 'run_centralized_baseline:=false', '-p', 'run_explicit_anchor_centralized_baseline:=false'])
        feeder = Node('distributed_test_input')
        scheduler = DistributedCbsNode()
        scheduler.next_run = float('inf')
        displayed = set()
        report = scheduler._report_alignment
        def observe_iterations():
            values = [scheduler.alignments.get(r, {}) for r in robots]
            rounds = {v.get('iteration') for v in values if v.get('provisional')}
            if len(rounds) == 1 and all(v.get('provisional') for v in values):
                displayed.update(rounds)
            report()
        scheduler._report_alignment = observe_iterations
        executor = SingleThreadedExecutor()
        executor.add_node(scheduler); executor.add_node(feeder)
        if not remote:
            for robot in robots:
                agent = CbsAgent(use_global_arguments=False, parameter_overrides=[Parameter('robot_id',value=robot)])
                agents.append(agent); executor.add_node(agent)
        def until(check, seconds=180):
            end = time.monotonic()+seconds
            while not check():
                if time.monotonic()>end:raise AssertionError('Distributed test timed out')
                executor.spin_once(timeout_sec=.05)
        try:
            until(lambda: set(robots) <= scheduler._available_agents(), 40)
            pubs = []
            session = f'{time.time_ns():020d}-distributed-test'
            for robot in robots:
                pub = feeder.create_publisher(RosPath, f'/{robot}/dpvo/path',10); pubs.append(pub)
                until(lambda: pub.get_subscription_count()>0,10)
                path = RosPath();path.header.frame_id=f'{robot}/{session}/map'
                for j in range(3):
                    pose=PoseStamped();pose.pose.position.x=j*.5;pose.pose.orientation.w=1.;path.poses.append(pose)
                pub.publish(path)
            until(lambda: all(r in scheduler.paths for r in robots))
            pub = feeder.create_publisher(InterRobotLoopClosure,'/dpvo_multi_robot/loop_closure',50)
            until(lambda:pub.get_subscription_count()>0,10)
            for a,b in zip(robots,robots[1:]):
                msg=InterRobotLoopClosure(query_robot_id=a,match_robot_id=b,
                    query_session_id=session,match_session_id=session,query_pose=[0.,0.,0.,0.,0.,0.,1.],
                    match_pose=[0.,0.,0.,0.,0.,0.,1.],scale=1.,inliers=50,inlier_ratio=1.,verification_method='distributed-smoke')
                msg.query_to_match.rotation.w=1.;pub.publish(msg)
            until(lambda: len(scheduler.constraints) == count-1)
            scheduler.next_run = 0.
            until(lambda: (Path(directory)/'distributed_latest.json').exists())
            assert not displayed, displayed
            assert not any(v.get('provisional') for v in scheduler.alignments.values())
            print('Displayed only the completed cycle', flush=True)
            audit=json.loads((Path(directory)/'distributed_latest.json').read_text())
            assert all(v['optimizer_instances']==1 and v['sent']>0 and v['received']>0 for v in audit['results'].values())
            if remote:
                assert len({v['host'] for v in audit['results'].values()}) == count
            print(json.dumps(audit,indent=2),flush=True)
        finally:
            scheduler.close()
            for a in agents:a.close()
            executor.shutdown()
            for n in [scheduler,feeder]+agents:n.destroy_node()
            rclpy.shutdown()


if __name__=='__main__':main()

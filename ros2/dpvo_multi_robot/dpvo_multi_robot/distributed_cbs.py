"""Snapshot scheduling and display aggregation; no optimizer runs in this node."""
import json
from pathlib import Path
import time
import uuid

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

from dpvo.loop_closure.centralized import CentralizedPgoResult
from dpvo.loop_closure.pose_graph import build_keyframe_graph
from .online_cbs import OnlineCbsNode
from .online_common import robot_components


class DistributedCbsNode(OnlineCbsNode):
    def __init__(self):
        self.agents = {}
        self.results = {}
        self.dispatch_closed = False
        self.current_epoch = None
        self.current_job = None
        super().__init__()
        if self.get_parameter('stage_mode').value != 'alternating':
            raise ValueError('The network CBS runtime currently requires alternating stages')
        self.job_pub = self.create_publisher(String, '/dpvo_multi_robot/cbs/jobs',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.release_pub = self.create_publisher(String, '/dpvo_multi_robot/cbs/release', 20)
        self.create_subscription(String, '/dpvo_multi_robot/cbs/agent_status', self.agent_status, 50)
        self.create_subscription(String, '/dpvo_multi_robot/cbs/agent_results', self.agent_result, 50)

    def agent_status(self, message):
        try:
            value = json.loads(message.data)
            robot = value['robot']
            if robot not in self.robot_ids:
                return
            with self.lock:
                self.agents[robot] = (time.monotonic(), value)
        except (ValueError, KeyError, TypeError):
            pass

    def agent_result(self, message):
        try:
            value = json.loads(message.data)
            with self.lock:
                if value.get('epoch') == self.current_epoch and value['robot'] in self.robot_ids:
                    # Ignore previews from older agents during rolling upgrades.
                    if not value.get('provisional'):
                        self.results[value['robot']] = value
        except (ValueError, KeyError, TypeError):
            pass

    def _available_agents(self):
        now = time.monotonic()
        return {r for r, (seen, _) in self.agents.items() if now - seen < 5}

    def _signature(self):
        return super()._signature(), tuple(sorted(
            (r, self.agents[r][1].get('boot')) for r in self._available_agents()))

    def _ready(self, require_done=False):
        available = self._available_agents()
        members, _ = self._eligible(self.constraints,
            {r: p.poses for r, p in self.paths.items() if r in available}, self.active_sessions)
        return (True, 'ready') if members else (False, 'waiting for connected maps and at least two live CBS agents')

    def _wait(self, job, predicate, timeout, release=False):
        end = time.monotonic() + timeout
        retry = 0.
        while not self.dispatch_closed and time.monotonic() < end:
            with self.lock:
                if any(self.active_sessions.get(r) != s for r, s in job['sessions'].items()):
                    raise RuntimeError('A robot restarted during this CBS epoch')
                for robot in job['sessions']:
                    status = self.agents.get(robot)
                    if status is None or time.monotonic() - status[0] > 8:
                        raise RuntimeError(f'CBS agent {robot} disconnected')
                    if status[1].get('epoch') == job['epoch'] and status[1].get('phase') == 'error':
                        raise RuntimeError(f"{robot}: {status[1].get('error')}")
                    result = self.results.get(robot)
                    if result is not None and not result.get('ok'):
                        raise RuntimeError(f"{robot}: {result.get('error')}")
                if predicate():
                    return
            if time.monotonic() > retry:
                if not release:
                    self.job_pub.publish(String(data=json.dumps(job, allow_nan=False)))
                if release:
                    self.release_pub.publish(String(data=json.dumps({'epoch': job['epoch']})))
                retry = time.monotonic() + 1
            time.sleep(.05)
        raise RuntimeError('Distributed CBS epoch timed out or cancelled')

    def _validate_results(self, selected, results=None):
        """Called under self.lock immediately before committing the whole group."""
        if self.dispatch_closed or any(self.active_sessions.get(r) != session
                                       for r, session in selected.items()):
            raise RuntimeError('A robot restarted before CBS results were committed')
        results = self.results if results is None else results
        for robot in selected:
            result = results[robot]
            if result.get('sessions') != selected or result.get('optimizer_instances') != 1:
                raise RuntimeError('Incompatible agent result')
            values = result['translation'] + result['quaternion'] + [result['scale']]
            if (len(result['translation']) != 3 or len(result['quaternion']) != 4
                    or not np.isfinite(values).all() or result['scale'] <= 0
                    or np.linalg.norm(result['quaternion']) < 1e-8):
                raise RuntimeError('Invalid agent map transform')

    def _run(self, constraints, paths, sessions, online, timestamps=None):
        with self.lock:
            live = self._available_agents()
        members, constraints = self._eligible(constraints,
            {r: p for r, p in paths.items() if r in live}, sessions)
        groups = robot_components(members, [(c.query_robot[0], c.match_robot[0]) for c in constraints])
        for group in groups:
            if len(group) < 2 or self.dispatch_closed:
                continue
            group_constraints = [c for c in constraints if c.query_robot[0] in group and c.match_robot[0] in group]
            selected = {r: sessions[r] for r in group}
            graph = build_keyframe_graph(group_constraints, CentralizedPgoResult({}, True, 0., 0.),
                {r: paths[r] for r in group}, selected,
                self.get_parameter('anchor_robot_id').value,
                odometry_weight=float(self.get_parameter('pose_graph_odometry_weight').value),
                align_to_global=False, timestamps={r: timestamps[r] for r in group} if timestamps else None)
            if graph.metadata['skipped_inter_robot_loops']:
                self.get_logger().warning('Waiting for paths containing all verified keyframes')
                continue
            job = dict(epoch=uuid.uuid4().hex, sessions=selected, graph=graph.to_dict(),
                iterations=int(self.get_parameter('iterations').value), expires=time.time() + 300,
                warmup=int(self.get_parameter('pose_warmup_iterations').value),
                pose_block=int(self.get_parameter('pose_block_iterations').value),
                anchor_block=int(self.get_parameter('anchor_block_iterations').value),
                parameters={p: float(self.get_parameter(p).value) for p in
                    ('target_hellinger', 'contract_alpha', 'd_reset')})
            with self.lock:
                self.current_epoch, self.results = job['epoch'], {}
                self.current_job = job
                previous = {r: self.alignments.get(r) for r in selected}
            try:
                self.job_pub.publish(String(data=json.dumps(job, allow_nan=False)))
                self._wait(job, lambda: all(self.agents[r][1].get('epoch') == job['epoch']
                    and self.agents[r][1].get('phase') == 'ready' for r in group), 60)
                self.get_logger().info(f"Distributed CBS epoch {job['epoch'][:8]}: agents {group}")
                self.release_pub.publish(String(data=json.dumps({'epoch': job['epoch']})))
                self._wait(job, lambda: all(r in self.results for r in group), 240, release=True)
                with self.lock:
                    self._validate_results(selected)
                    self.current_job = None
                    # Atomic commit only when every member finished the same snapshot.
                    for robot in group:
                        result = self.results[robot]
                        self.alignments[robot] = dict(sessions=selected,
                            translation=result['translation'], quaternion=result['quaternion'],
                            scale=result['scale'], constraints=len(group_constraints))
                    self._report_alignment()
                    output = Path(self.get_parameter('output_dir').value)
                    output.mkdir(parents=True, exist_ok=True)
                    audit = dict(epoch=job['epoch'], sessions=selected, results=dict(self.results),
                                 distributed=True, iterations=job['iterations'])
                    (output / 'distributed_latest.json').write_text(json.dumps(audit, indent=2))
                self.get_logger().info(f"Distributed CBS epoch complete: {group}")
            except Exception as error:
                with self.lock:
                    self.current_job = None
                    for robot, alignment in previous.items():
                        if alignment is None:
                            self.alignments.pop(robot, None)
                        else:
                            self.alignments[robot] = alignment
                    self._report_alignment()
                self.get_logger().error(f'Distributed CBS: {error}')
            finally:
                self.release_pub.publish(String(data=json.dumps({'epoch': job['epoch'], 'cancel': True})))
                with self.lock:
                    self.current_epoch = None
                    self.current_job = None

    def close(self):
        self.dispatch_closed = True
        super().close()


def main():
    rclpy.init()
    node = DistributedCbsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

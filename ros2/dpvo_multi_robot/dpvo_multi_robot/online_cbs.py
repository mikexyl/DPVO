"""Periodically optimize immutable live keyframe snapshots with the existing CBS solver."""
import time
import json
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy

import rclpy

from .cbs_pgo import CbsPgoNode
from .online_common import robot_components, fleet_frame, parse_session_frame


class OnlineCbsNode(CbsPgoNode):
    def __init__(self):
        self.revision = 0
        self.next_run = 0.0
        self.alignments = {}
        super().__init__()
        self.alignment_publisher = self.create_publisher(String,
            '/dpvo_multi_robot/cbs/alignment_status',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter('online_period', 10.0)

    def _activate_session(self, robot_id, session_id):
        if robot_id not in self.robot_ids:
            return False
        previous = self.active_sessions.get(robot_id)
        accepted = super()._activate_session(robot_id, session_id)
        if accepted and previous != session_id:
            self.paths.pop(robot_id, None)
            self.done_robots.discard(robot_id)
            self.revision += 1
        return accepted

    def _path(self, robot_id, message):
        try:
            robot, session = parse_session_frame(message.header.frame_id)
        except ValueError:
            return
        if robot != robot_id:
            return
        with self.lock:
            if not self._activate_session(robot, session):
                return
            super()._path(robot_id, message)
            self.revision += 1

    def _signature(self):
        # BA can change old poses without changing the number of keyframes.
        return super()._signature(), self.revision

    @staticmethod
    def _eligible(constraints, paths, sessions):
        available = {r for r, path in paths.items() if path and r in sessions}
        constraints = [c for c in constraints
            if c.query_robot[0] in available and c.match_robot[0] in available
            and sessions[c.query_robot[0]] == c.query_robot[1]
            and sessions[c.match_robot[0]] == c.match_robot[1]]
        groups = robot_components(available,
            [(c.query_robot[0], c.match_robot[0]) for c in constraints])
        members = {r for group in groups if len(group) > 1 for r in group}
        return members, constraints

    def _ready(self, require_done=False):
        members, _ = self._eligible(self.constraints,
            {r: p.poses for r, p in self.paths.items()}, self.active_sessions)
        return (True, 'ready') if members else (False, 'waiting for a verified connection between any two robot maps')

    def _run(self, constraints, paths, sessions, online, timestamps=None):
        members, constraints = self._eligible(constraints, paths, sessions)
        if not members:
            return
        constraints = [c for c in constraints if c.query_robot[0] in members
                       and c.match_robot[0] in members]
        super()._run(constraints, {r: paths[r] for r in members},
            {r: sessions[r] for r in members}, online,
            {r: timestamps[r] for r in members} if timestamps is not None else None)

    def _report_alignment(self):
        # Preserve a stopped/disconnected robot's map, but never reuse a restarted session.
        valid = {r: value for r, value in self.alignments.items()
                 if all(self.active_sessions.get(peer) == session
                        for peer, session in value['sessions'].items())}
        self.alignments = valid
        self.alignment_publisher.publish(String(data=json.dumps({
            'sessions': self.active_sessions, 'alignments': valid,
            'connected_groups': [g for g in robot_components(self.active_sessions,
                [(c.query_robot[0], c.match_robot[0]) for c in self.constraints]) if len(g) > 1]})))

    def _timer(self):
        now = time.monotonic()
        with self.lock:
            self._report_alignment()
            if now < self.next_run or self._signature() == self.last_signature:
                return
            launched, reason = self._launch(force=True)
            if launched:
                self.next_run = now + max(1.0, float(self.get_parameter('online_period').value))
                self.get_logger().info(reason)

    def _result_frame(self):
        return fleet_frame(self._current_group)

    def _publish_results(self, graph, output_dir, online, duration):
        with self.lock:
            snapshot = {vertex.robot_id: vertex.session_id for vertex in graph.vertices}
            if any(self.active_sessions.get(robot) != session for robot, session in snapshot.items()):
                self.get_logger().warning('Discarding CBS result from a superseded robot session')
                return
            vertices = {v.vertex_id: v.robot_id for v in graph.vertices}
            groups = robot_components(snapshot, [(vertices[e.source], vertices[e.target])
                                      for e in graph.edges])
            self._result_groups = {r: {p: snapshot[p] for p in group}
                                   for group in groups if len(group) > 1 for r in group}
            self._pending_alignments = {}
            super()._publish_results(graph, output_dir, online, duration)
            self.alignments = self._pending_alignments
            self._report_alignment()

    def _publish_transform(self, key, transform, count, cost, residual_norm):
        robot, session = key
        self._current_group = self._result_groups[robot]
        self._pending_alignments[robot] = dict(sessions=self._current_group,
            translation=transform.translation.tolist(),
            quaternion=Rotation.from_matrix(transform.rotation.copy()).as_quat().tolist(),
            scale=float(transform.scale), constraints=int(count))
        super()._publish_transform(key, transform, count, cost, residual_norm)


def main(args=None):
    rclpy.init(args=args)
    node = OnlineCbsNode()
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

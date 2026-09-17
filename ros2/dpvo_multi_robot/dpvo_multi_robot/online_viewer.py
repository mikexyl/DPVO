"""Phone-friendly fleet controls and session-safe shared Sim(3) map display."""
import json
import io
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, CompressedImage
from PIL import Image as PillowImage
from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool
import viser

from dpvo_multi_robot_interfaces.msg import RobotMapTransform
from .online_common import fleet_frame, parse_session_frame
from .online_status import health_text, compact_health_text


class FleetViewer(Node):
    def __init__(self, **node_options):
        super().__init__('dpvo_fleet_viewer', **node_options)
        self.declare_parameter('robot_ids', ['robot0', 'robot1'])
        self.declare_parameter('web_port', 9091)
        self.declare_parameter('transform_topic', '/dpvo_multi_robot/cbs/map_transforms')
        self.robots = list(self.get_parameter('robot_ids').value)
        self.lock = threading.RLock()
        self.server = viser.ViserServer(host='0.0.0.0',
            port=self.get_parameter('web_port').value,
            label='DPVO Fleet' if len(self.robots) > 1 else f'DPVO {self.robots[0]}')
        self.server.scene.set_up_direction('-y')
        self.server.gui.configure_theme(control_layout='collapsible', control_width='small')
        self.states = {}
        self.alignment_labels = {}
        self.alignment_sessions = {}
        self.alignment_groups = {}
        self.alignment_seen = None
        self.agent_labels = {}
        self.agent_seen = {}
        self.agent_statuses = {}
        self.commands = []
        self.control_pending = {}
        self.control_clients = {}
        self.control_buttons = {}
        self.statuses = {}
        self.dense_controls = {}
        self.last_seen = {}
        self.health = {}
        self.health_labels = {}
        self.health_details = {}
        self.previews = {}
        self.preview_labels = {}
        self.preview_seen = {}
        with self.server.gui.add_folder('Map display', expand_by_default=False):
            self.size = self.server.gui.add_slider('Point size', min=.001, max=.05,
                                                   step=.001, initial_value=.005)
            self.dense_size = self.server.gui.add_slider('Dense point size', min=.001, max=.05, step=.001, initial_value=.005)
        @self.size.on_update
        def point_size(_):
            with self.lock:
                for state in self.states.values():
                    state['cloud'].point_size = self.size.value
        @self.dense_size.on_update
        def dense_size(_):
            with self.lock:
                for state in self.states.values():
                    state['dense'].point_size = self.dense_size.value
        colors = [(0, 160, 255), (255, 140, 0), (100, 210, 100), (190, 100, 240)]
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscriptions_owned = []
        self.camera_panel = self.server.gui.add_panel()
        camera_tab = self.camera_panel.add_tab('Robot cameras')
        self.camera_panel.dock_left()
        self.camera_panel.set_width(480)
        for index, robot in enumerate(self.robots):
            with self.server.gui.add_folder(robot):
                self.health_labels[robot] = self.server.gui.add_markdown('⚪ Waiting')
                self.alignment_labels[robot] = self.server.gui.add_markdown('⚪ Unmerged')
                for action in ('start', 'stop'):
                    button = self.server.gui.add_button(action.title(), disabled=True,
                        color='green' if action == 'start' else 'red',
                        hint=f'{action.title()} camera and tracker')
                    self.control_buttons[robot, action] = button
                    button.on_click(lambda _, r=robot, a=action: self.queue_command(r, a))
                    self.control_clients[robot, action] = self.create_client(Trigger, f'/{robot}/dpvo/{action}')
                with self.server.gui.add_folder('Options & details', expand_by_default=False):
                    toggle = self.server.gui.add_checkbox('Dense mapping', initial_value=False)
                    toggle.on_update(lambda event, r=robot: self.queue_command(r, ('dense', event.target.value)) if event.client is not None else None)
                    self.dense_controls[robot] = toggle
                    self.control_clients[robot, 'dense'] = self.create_client(SetBool, f'/{robot}/dpvo/set_dense_mapping')
                    self.health_details[robot] = self.server.gui.add_markdown('Waiting for heartbeat')
                    self.agent_labels[robot] = self.server.gui.add_markdown('CBS agent: waiting')
                # Only control errors occupy space outside the collapsed details.
                self.statuses[robot] = self.server.gui.add_markdown('', visible=False)
            with camera_tab:
                with self.server.gui.add_folder(f'{robot} camera'):
                    self.previews[robot] = self.server.gui.add_image(
                        np.zeros((240, 384, 3), np.uint8), label=None,
                        format='jpeg', jpeg_quality=75)
                    self.preview_labels[robot] = self.server.gui.add_markdown('⚪ Camera off')
            # Unaligned maps are separated and labelled; never imply a shared origin.
            offset = np.array([index * 5., 0., 0.])
            root = f'/robots/{robot}'
            frame = self.server.scene.add_frame(root, show_axes=True, position=offset)
            label = self.server.scene.add_label(root + '/label', f'{robot}: unaligned')
            cloud = self.server.scene.add_point_cloud(root + '/points',
                points=np.zeros((1, 3), dtype=np.float32), colors=colors[index % len(colors)],
                point_size=self.size.value, precision='float32', visible=False)
            dense = self.server.scene.add_point_cloud(root + '/dense',
                points=np.zeros((1, 3), dtype=np.float32), colors=(180, 180, 180),
                point_size=self.dense_size.value, precision='float32', visible=False)
            trajectory = self.server.scene.add_line_segments(root + '/path',
                points=np.zeros((1, 2, 3), dtype=np.float32), colors=colors[index % len(colors)],
                thickness=2, thickness_units='screen', visible=False)
            self.states[robot] = dict(session=None, transform=None, frame=frame, label=label,
                                      cloud=cloud, dense=dense, raw_dense=np.empty((0, 3), dtype=np.float32), path=trajectory, offset=offset, scale=1.,
                                      raw_points=np.empty((0, 3), dtype=np.float32),
                                      raw_path=np.empty((0, 2, 3), dtype=np.float32))
            self.subscriptions_owned.extend([
                self.create_subscription(Path, f'/{robot}/dpvo/path',
                    lambda m, r=robot: self.path(r, m), 1),
                self.create_subscription(PointCloud2, f'/{robot}/dpvo/points',
                    lambda m, r=robot: self.points(r, m),
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)),
                self.create_subscription(PointCloud2, f'/{robot}/dpvo/dense_points',
                    lambda m, r=robot: self.points(r, m, dense=True),
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)),
                self.create_subscription(CompressedImage, f'/{robot}/dpvo/preview/compressed',
                    lambda m, r=robot: self.preview(r, m),
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)),
                self.create_subscription(String, f'/{robot}/dpvo/control_status',
                    lambda m, r=robot: self.status(r, m), qos),
            ])
        self.create_subscription(RobotMapTransform, self.get_parameter('transform_topic').value,
                                 self.transform, 50)
        self.create_subscription(String, '/dpvo_multi_robot/cbs/agent_status', self.cbs_agent_status, 50)
        self.create_subscription(String, '/dpvo_multi_robot/cbs/alignment_status',
                                 self.alignment_status, qos)
        self.create_timer(.2, self.tick)
        @self.server.on_client_connect
        def connected(client):
            center = 2.5 * (len(self.robots) - 1)
            distance = max(6., 6. * (len(self.robots) - 1))
            client.camera.position = (center, -distance / 2., -distance)
            client.camera.look_at = (center, 0., 0.)
            client.camera.up_direction = (0., -1., 0.)

    def queue_command(self, robot, action):
        with self.lock:
            self.commands.append((robot, action))

    def tick(self):
        with self.lock:
            for robot, (future, deadline) in list(self.control_pending.items()):
                if time.monotonic() > deadline:
                    del self.control_pending[robot]
                    future.cancel()
                    self.statuses[robot].content = '⚠ Control timed out. Retry when online.'
                    self.statuses[robot].visible = True
            commands, self.commands = self.commands, []
        for robot, action in commands:
            with self.lock:
                if robot in self.control_pending:
                    self.commands.append((robot, action))
                    continue
            dense = isinstance(action, tuple)
            client = self.control_clients[robot, 'dense' if dense else action]
            if not client.service_is_ready():
                self.statuses[robot].content = '⚠ Robot unavailable.'
                self.statuses[robot].visible = True
                continue
            self.statuses[robot].visible = False
            future = client.call_async(SetBool.Request(data=action[1]) if dense else Trigger.Request())
            with self.lock:
                self.control_pending[robot] = (future, time.monotonic() + 15.)
            def done(result, r=robot):
                try:
                    response = result.result()
                    self.statuses[r].content = response.message
                    self.statuses[r].visible = not response.success
                except Exception as error:
                    self.statuses[r].content = f'Control request failed: {error}'
                    self.statuses[r].visible = True
                finally:
                    with self.lock:
                        pending = self.control_pending.get(r)
                        if pending is not None and pending[0] is result:
                            del self.control_pending[r]
            future.add_done_callback(done)
        now = time.monotonic()
        for robot in getattr(self, 'robots', []):
            self.refresh_controls(robot, now)
            self.health_labels[robot].content = compact_health_text(self.health.get(robot), now - self.last_seen.get(robot, now))
            self.health_details[robot].content = health_text(self.health.get(robot), now - self.last_seen.get(robot, now))
            if robot in getattr(self, 'agent_labels', {}):
                age = now - self.agent_seen.get(robot, -float('inf'))
                value = self.agent_statuses.get(robot, {})
                phase = value.get('phase', 'waiting')
                detail = f" · round {value.get('round', 0)} · received {value.get('received', 0)}" if phase in ('optimizing', 'complete') else ''
                self.agent_labels[robot].content = (f"CBS agent: {phase}{detail}" if age < 5 else 'CBS agent: offline')
            seen = getattr(self, 'alignment_seen', None)
            if seen is not None and now - seen > 5 and robot in self.alignment_labels:
                state = self.states[robot]
                self.alignment_labels[robot].content = ('🟡 Merge stale'
                    if state.get('merged_sessions') else '⚪ Unmerged · CBS offline')
            last = self.preview_seen.get(robot)
            if last is not None:
                age = now - last
                self.preview_labels[robot].content = ('🟢 Live' if age < 3
                    else f'🟡 Stale · {age:.0f}s')

    def refresh_controls(self, robot, now=None):
        now = time.monotonic() if now is None else now
        status = self.health.get(robot)
        connected = status is not None and now - self.last_seen.get(robot, -float('inf')) <= 3
        pending = robot in self.control_pending
        running = bool(status and status.get('state') == 'worker_running')
        self.control_buttons[robot, 'start'].disabled = not connected or pending or running
        self.control_buttons[robot, 'stop'].disabled = not connected or pending or not running
        self.dense_controls[robot].disabled = not connected or pending or running

    def status(self, robot, message):
        try:
            status = json.loads(message.data)
        except ValueError:
            return
        if not isinstance(status, dict):
            return
        self.last_seen[robot] = time.monotonic()
        previous = self.health.get(robot, {})
        self.health[robot] = status
        if status.get('state') != 'worker_running' and (previous.get('state') != status.get('state') or robot in self.preview_seen):
            self.previews[robot].image = np.zeros((240, 384, 3), np.uint8)
            self.preview_seen.pop(robot, None)
            self.preview_labels[robot].content = 'Camera off'
        if status.get('state') == 'worker_running' and robot not in self.preview_seen:
            self.preview_labels[robot].content = '🟡 Starting camera…'
        self.refresh_controls(robot)
        self.dense_controls[robot].value = bool(status.get('dense_enabled', False))

    def preview(self, robot, message):
        try:
            identity, session = parse_session_frame(message.header.frame_id)
            if identity != robot or len(message.data) > 500000:
                return
            if self.health.get(robot, {}).get('state') == 'stopped':
                return
            with PillowImage.open(io.BytesIO(bytes(message.data))) as image:
                if image.width > 1280 or image.height > 800:
                    return
                rgb = np.asarray(image.convert('RGB')).copy()
        except (ValueError, OSError):
            return
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        with self.lock:
            if not self.activate(robot, session):
                return
            state = self.states[robot]
            if stamp <= state.get('preview_stamp', -1):
                return
            state['preview_stamp'] = stamp
            self.previews[robot].image = rgb
            self.preview_seen[robot] = time.monotonic()

    def activate(self, robot, session):
        state = self.states[robot]
        current = state['session']
        if current is not None and session < current:
            return False
        if current != session:
            # Only maps sharing the restarted robot's alignment lose their transform.
            for other_robot, other in self.states.items():
                if other_robot == robot or robot in other.get('merged_sessions', {}):
                    self.reset_alignment(other_robot)
            state['session'], state['transform'] = session, None
            state['preview_stamp'] = -1
            if hasattr(self, 'previews'):
                self.previews[robot].image = np.zeros((240, 384, 3), np.uint8)
                self.preview_seen.pop(robot, None)
            state['cloud'].visible = state['path'].visible = False
            state['frame'].position = state['offset']
            state['frame'].wxyz = (1., 0., 0., 0.)
            state['scale'] = 1.
            if 'dense' in state:
                state['dense'].visible = False
                state['raw_dense'] = np.empty((0, 3), dtype=np.float32)
            state['raw_points'] = np.empty((0, 3), dtype=np.float32)
            state['raw_path'] = np.empty((0, 2, 3), dtype=np.float32)
            state['label'].text = f'{robot}: unaligned'
        return True

    def path(self, robot, message):
        try:
            identity, session = parse_session_frame(message.header.frame_id)
        except ValueError:
            return
        if identity != robot:
            return
        with self.lock:
            if not self.activate(robot, session):
                return
            points = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                               for p in message.poses], dtype=np.float32)
            if len(points) > 1:
                state = self.states[robot]
                state['raw_path'] = np.stack([points[:-1], points[1:]], axis=1)
                state['path'].points = state['raw_path'] * state['scale']
                self.states[robot]['path'].visible = True

    def points(self, robot, message, dense=False):
        try:
            identity, session = parse_session_frame(message.header.frame_id)
        except ValueError:
            return
        if (identity != robot or message.point_step != 16 or message.is_bigendian
                or message.height != 1 or message.row_step != message.width * 16
                or len(message.data) != message.width * 16):
            return
        with self.lock:
            if not self.activate(robot, session):
                return
            dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
            packed = np.frombuffer(bytes(message.data), dtype=dtype)
            points = np.column_stack([packed['x'], packed['y'], packed['z']])
            rgb = packed['rgb']
            colors = np.column_stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255]).astype(np.uint8)
            valid = np.isfinite(points).all(axis=1)
            state = self.states[robot]
            raw, handle = ('raw_dense', 'dense') if dense else ('raw_points', 'cloud')
            state[raw] = points[valid].astype(np.float32)
            state[handle].points = state[raw] * state['scale']
            state[handle].colors = colors[valid]
            state[handle].visible = bool(valid.any())

    def cbs_agent_status(self, message):
        try:
            value = json.loads(message.data)
            robot = value['robot']
            if robot in self.states:
                self.agent_statuses[robot] = value
                self.agent_seen[robot] = time.monotonic()
        except (ValueError, KeyError, TypeError):
            pass

    def reset_alignment(self, robot):
        state = self.states[robot]
        state['frame'].position = state['offset']
        state['frame'].wxyz = (1., 0., 0., 0.)
        state['scale'] = 1.
        state['merged_sessions'] = {}
        state['cloud'].points = state.get('raw_points', np.empty((0, 3), np.float32))
        state['path'].points = state.get('raw_path', np.empty((0, 2, 3), np.float32))
        if 'dense' in state:
            state['dense'].points = state.get('raw_dense', np.empty((0, 3), np.float32))
        state['label'].text = f'{robot}: unmerged'
        if robot in getattr(self, 'alignment_labels', {}):
            self.alignment_labels[robot].content = '⚪ Unmerged'

    def alignment_status(self, message):
        try:
            data = json.loads(message.data)
            sessions, alignments = data['sessions'], data['alignments']
            if not isinstance(sessions, dict) or not isinstance(alignments, dict):
                return
        except (ValueError, KeyError, TypeError):
            return
        with self.lock:
            self.alignment_seen = time.monotonic()
            # Do not let a delayed coordinator snapshot overwrite a locally newer session.
            for robot, session in sessions.items():
                if session >= self.alignment_sessions.get(robot, ''):
                    self.alignment_sessions[robot] = session
            self.alignment_groups = {fleet_frame(v['sessions']): v['sessions'] for v in alignments.values()}
            for robot, state in self.states.items():
                value = alignments.get(robot)
                group = value['sessions'] if value else {}
                valid = (value is not None and len(group) > 1
                         and group.get(robot) == state['session']
                         and all(self.alignment_sessions.get(p) == v for p, v in group.items()))
                if not valid:
                    self.reset_alignment(robot)
                    connected = any(robot in g for g in data.get('connected_groups', []))
                    self.alignment_labels[robot].content = ('🟡 Aligning'
                        if connected else '⚪ Unmerged')
                    continue
                msg = RobotMapTransform(robot_id=robot, session_id=state['session'], scale=value['scale'])
                msg.header.frame_id = fleet_frame(group)
                t, q = msg.local_map_to_global.translation, msg.local_map_to_global.rotation
                t.x, t.y, t.z = map(float, value['translation'])
                q.x, q.y, q.z, q.w = map(float, value['quaternion'])
                self.transform(msg)
                if value.get('provisional'):
                    self.alignment_labels[robot].content = (
                        f"🟡 CBS · {value['iteration']}/{value['iterations']}")
                    state['label'].text = f"{robot}: CBS {value['iteration']}/{value['iterations']}"

    def transform(self, message):
        robot = message.robot_id
        if robot not in self.states:
            return
        with self.lock:
            state = self.states[robot]
            # Transforms are accepted only for the map actually displayed.
            sessions = getattr(self, 'alignment_groups', {}).get(message.header.frame_id)
            if sessions is None:
                # Legacy full-fleet transform support for older coordinators.
                sessions = {r: s['session'] for r, s in self.states.items()}
                if any(v is None for v in sessions.values()) or message.header.frame_id != fleet_frame(sessions):
                    return
            if message.session_id != state['session'] or sessions.get(robot) != state['session']:
                return
            for peer, session in sessions.items():
                known = self.states.get(peer, {}).get('session')
                announced = getattr(self, 'alignment_sessions', {}).get(peer)
                if (known is not None and known != session) or (announced is not None and announced != session):
                    return
            value = message.local_map_to_global
            position = (value.translation.x, value.translation.y, value.translation.z)
            quaternion = np.array([value.rotation.w, value.rotation.x, value.rotation.y, value.rotation.z])
            if (not np.isfinite(message.scale) or message.scale <= 0
                    or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                    or np.linalg.norm(quaternion) < 1e-8):
                return
            # Keep separately merged groups apart until a verified bridge joins them.
            anchor = min(sessions)
            offset = self.states.get(anchor, {}).get('offset', np.zeros(3))
            state['frame'].position = tuple(np.asarray(position) + offset)
            state['merged_sessions'] = dict(sessions)
            state['frame'].wxyz = quaternion / np.linalg.norm(quaternion)
            state['scale'] = message.scale
            state['cloud'].points = state['raw_points'] * message.scale
            if 'dense' in state:
                state['dense'].points = state['raw_dense'] * message.scale
            state['path'].points = state['raw_path'] * message.scale
            state['label'].text = f'{robot}: CBS aligned'
            if robot in getattr(self, 'alignment_labels', {}):
                peers = ', '.join(p for p in sorted(sessions) if p != robot)
                self.alignment_labels[robot].content = f'🟢 Merged with {peers}'


def main(args=None):
    rclpy.init(args=args)
    node = FleetViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.server.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

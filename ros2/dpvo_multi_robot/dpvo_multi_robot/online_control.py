"""Idle ROS control endpoint: launches a fresh camera/tracker process on Start."""
import json
from pathlib import Path
import sys
import time
import socket
import uuid
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String, Header
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import Trigger, SetBool

from .online_common import WorkerProcess


class OnlineControl(Node):
    def __init__(self):
        super().__init__('dpvo_control')
        self.declare_parameter('robot_id', 'robot0')
        self.declare_parameter('worker_parameters', '')
        robot = self.get_parameter('robot_id').value
        path = Path(self.get_parameter('worker_parameters').value)
        if not path.is_file():
            raise ValueError(f'Worker parameter file does not exist: {path}')
        parameters = yaml.safe_load(path.read_text())
        tracker = parameters.get(f'/{robot}/dpvo_multi_robot', {}).get('ros__parameters', {})
        self.dense_engine = Path(tracker.get('dense_engine', '/output/da3-small-238x378/da3.engine'))
        self.started_at = time.monotonic()
        self.boot_id = uuid.uuid4().hex
        self.heartbeat = 0
        self.camera_seen = self.tracker_seen = None
        self.tracking_status = {}
        self.create_subscription(CameraInfo, 'camera/camera_info', self.camera_tick, 1)
        self.create_subscription(Header, 'dpvo/frame_ack', self.tracker_tick, 1)
        self.create_subscription(String, 'dpvo/tracking_status', self.tracking_tick, 1)
        self.dense_enabled = False
        self.worker = WorkerProcess([
            sys.executable, '-m', 'dpvo_multi_robot.online_worker', '--ros-args',
            '--params-file', str(path), '-r', f'__ns:=/{robot}',
        ])
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status = self.create_publisher(String, 'dpvo/control_status', qos)
        self.create_service(Trigger, 'dpvo/start', self.start)
        self.create_service(Trigger, 'dpvo/stop', self.stop)
        self.create_service(SetBool, 'dpvo/set_dense_mapping', self.set_dense)
        self.create_timer(0.5, self.report)
        self.report()

    def camera_tick(self, message):
        self.camera_seen = time.monotonic()

    def tracker_tick(self, message):
        self.tracker_seen = time.monotonic()

    def tracking_tick(self, message):
        try:
            self.tracking_status = json.loads(message.data)
        except ValueError:
            pass

    def report(self):
        running = self.worker.running
        now = time.monotonic()
        self.heartbeat += 1
        self.status.publish(String(data=json.dumps({
            'state': 'worker_running' if running else 'stopped',
            'exit_code': self.worker.last_exit,
            'dense_enabled': self.dense_enabled,
            'heartbeat': self.heartbeat, 'boot_id': self.boot_id, 'hostname': socket.gethostname(),
            'uptime_s': now - self.started_at,
            'camera_age_s': now - self.camera_seen if running and self.camera_seen is not None else None,
            'tracker_age_s': now - self.tracker_seen if running and self.tracker_seen is not None else None,
            'tracking': self.tracking_status if running else {},
        })))

    def set_dense(self, request, response):
        if self.worker.running:
            response.success = False
            response.message = 'Stop the robot before changing dense mapping'
        elif request.data and not (self.dense_engine.is_file() and self.dense_engine.with_name('manifest.json').is_file()):
            response.success = False
            response.message = 'DA3 TensorRT engine is not prepared on this robot'
        else:
            self.dense_enabled = bool(request.data)
            response.success = True
            response.message = 'Dense mapping ' + ('enabled for next Start' if request.data else 'disabled')
        self.report()
        return response

    def start(self, request, response):
        try:
            # Override applies to the disposable worker only; boot always starts disabled.
            if not self.worker.running:
                base = self.worker.command
                if '-p' in base:
                    base = base[:base.index('-p')]
                self.worker.command = base + ['-p', f'enable_dense_mapping:={str(self.dense_enabled).lower()}']
            if not self.worker.running:
                self.camera_seen = self.tracker_seen = None
                self.tracking_status = {}
            changed = self.worker.start()
            response.success = True
            response.message = 'Starting a fresh camera/tracker session' if changed else 'Already running'
        except Exception as error:
            response.success = False
            response.message = str(error)
        self.report()
        return response

    def stop(self, request, response):
        self.worker.stop()
        response.success = True
        response.message = 'Camera and tracker stopped'
        self.report()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = OnlineControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.worker.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

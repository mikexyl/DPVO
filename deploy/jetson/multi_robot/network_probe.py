"""Synthetic ROS probe for offline startup/reconnection; never opens a camera."""
import json
import os
from pathlib import Path
import sys
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Probe(Node):
    def __init__(self, robot, peer, output):
        super().__init__('network_probe_' + robot)
        self.output = Path(output)
        self.stats = dict(boot=uuid.uuid4().hex, pid=os.getpid(), local=0, remote=0,
                          services=0, sequence=0, last_remote=None)
        self.pub = self.create_publisher(String, f'/{robot}/dpvo/network_probe', 10)
        self.create_subscription(String, f'/{robot}/dpvo/network_probe', self.local, 10)
        self.create_subscription(String, f'/{peer}/dpvo/network_probe', self.remote, 10)
        self.create_service(Trigger, f'/{robot}/dpvo/network_probe_service', self.service)
        self.client = self.create_client(Trigger, f'/{peer}/dpvo/network_probe_service')
        self.pending = None
        self.deadline = 0.
        self.create_timer(.2, self.tick)

    def local(self, msg):
        self.stats['local'] += 1

    def remote(self, msg):
        self.stats['remote'] += 1
        self.stats['last_remote'] = time.time()

    def service(self, req, resp):
        resp.success = True
        resp.message = self.stats['boot']
        return resp

    def tick(self):
        self.stats['sequence'] += 1
        self.pub.publish(String(data=str(self.stats['sequence'])))
        if self.pending is not None and self.pending.done():
            if self.pending.result().success:
                self.stats['services'] += 1
            self.pending = None
        if self.pending is not None and time.monotonic() > self.deadline:
            self.client.remove_pending_request(self.pending)
            self.pending.cancel()
            self.pending = None
        if self.pending is None and self.client.service_is_ready():
            self.pending = self.client.call_async(Trigger.Request())
            self.deadline = time.monotonic() + 3.
        self.output.write_text(json.dumps(self.stats))


rclpy.init()
node = Probe(*sys.argv[1:4])
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()

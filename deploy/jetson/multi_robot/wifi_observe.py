"""Observe live ROS progress during a separately scheduled Wi-Fi outage.

Run inside a host-network ROS container. This observer does not control Wi-Fi,
start cameras, or restart any process. Arguments: robot interface seconds output.
"""
import json
from pathlib import Path
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

robot, interface, duration, output = sys.argv[1:]
rclpy.init()
node = Node('dpvo_wifi_observer')
result = {'samples': [], 'previews_online': 0, 'previews_offline': 0, 'boot_ids': [], 'sessions': []}
status = {}

def online():
    try:
        return (Path('/sys/class/net') / interface / 'carrier').read_text().strip() == '1'
    except OSError:
        return False

def receive(msg):
    status.update(json.loads(msg.data))

def preview(msg):
    result['previews_online' if online() else 'previews_offline'] += 1

node.create_subscription(String, f'/{robot}/dpvo/control_status', receive, 10)
node.create_subscription(CompressedImage, f'/{robot}/dpvo/preview/compressed', preview, qos_profile_sensor_data)
def record():
    sample = dict(at=time.monotonic(), wifi=online(), **status)
    result['samples'].append(sample)
    for key, value in [('boot_ids', status.get('boot_id')), ('sessions', status.get('tracking', {}).get('session'))]:
        if value and value not in result[key]:
            result[key].append(value)
    Path(output).write_text(json.dumps(result))
node.create_timer(1., record)
end = time.monotonic() + float(duration)
try:
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=.2)
finally:
    record()
    node.destroy_node()
    rclpy.shutdown()

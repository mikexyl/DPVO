"""Generate ROS parameters, local DDS discovery, and reconnecting fleet bridges."""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import yaml


class RosDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


PROFILES = Path(__file__).resolve().parent


def load_fleet(path):
    fleet = yaml.safe_load(Path(path).read_text())
    robots = fleet.get('robots', [])
    if not 2 <= len(robots) <= 4:
        raise ValueError('Configure two to four robots')
    ids = [r['id'] for r in robots]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', r) for r in ids):
        raise ValueError('Robot IDs must be unique ROS identifiers')
    addresses = [r['address'] for r in robots] + [fleet['coordinator']['address']]
    for address in addresses:
        ipaddress.IPv4Address(address)
    robot_addresses = [r['address'] for r in robots]
    if len(set(robot_addresses)) != len(robot_addresses):
        raise ValueError('Each robot must have its own IPv4 address')
    if not 0 <= int(fleet['ros_domain_id']) <= 232:
        raise ValueError('ROS domain must be between 0 and 232')
    if fleet['coordinator']['anchor_robot'] not in ids:
        raise ValueError('Anchor robot must belong to the fleet')
    if not 1024 <= int(fleet['coordinator']['web_port']) <= 65535:
        raise ValueError('Choose an unprivileged web port')
    port = int(fleet.get('network', {}).get('bridge_port', 17473))
    if fleet.get('network', {}).get('transport', 'zenoh') not in ('zenoh', 'cyclone'):
        raise ValueError('network.transport must be zenoh or cyclone')
    interface_address = fleet.get('network', {}).get('interface_address')
    if interface_address is not None:
        ipaddress.IPv4Address(interface_address)
    if not 1024 <= port <= 65535 or port in (9090, int(fleet['coordinator']['web_port'])):
        raise ValueError('Choose an unprivileged bridge_port distinct from panel ports')
    period = float(fleet['coordinator'].get('online_period', 10.0))
    if not math.isfinite(period) or period < 1:
        raise ValueError('online_period must be finite and at least one second')
    for robot in robots:
        if robot.get('camera_type', 'realsense') not in ('realsense', 'zed'):
            raise ValueError('camera_type must be realsense or zed')
        if robot.get('camera_type') == 'zed' and not str(robot.get('camera_serial', '')).isdigit():
            raise ValueError('ZED camera_serial must identify its factory calibration')
        if robot['profile'] not in ('jetpack62', 'jetpack72'):
            raise ValueError(f"Unknown JetPack profile: {robot['profile']}")
    return fleet


def generate(fleet, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = [r['id'] for r in fleet['robots']]
    def write(name, data):
        (output / name).write_text(yaml.dump(data, Dumper=RosDumper, sort_keys=False))
    for robot in fleet['robots']:
        profile = yaml.safe_load((PROFILES / f"{robot['profile']}.yaml").read_text())
        robot_id = robot['id']
        parameters = dict(
            robot_id=robot_id, session_id='', network='/models/dpvo.pth',
            config='/opt/dpvo/config/jetson_online.yaml', orb_vocab='/models/ORBvoc.txt',
            image_topic=f'/{robot_id}/camera/image_rect',
            camera_info_topic=f'/{robot_id}/camera/camera_info',
            image_scale=1.0, image_best_effort=True, session_frame_ids=True,
            viewer='none', trt_encoders='/output/engines-240x384',
            enable_dpvo_loop_closure=False, enable_inter_robot_loop_closure=True,
            retrieval_backend='megaloc', megaloc_repo='/models/loop_frontend/MegaLoc',
            megaloc_model_id='MegaLoc-5fe0dd697c4a70ba3e23607f6716ab3c606b16db',
            local_feature_backend='xfeat', xfeat_repo='/models/loop_frontend/accelerated_features',
            xfeat_top_k=1024, bow_repetitions=3, bow_nms_radius=30,
            teaser_required=True, min_inliers=int(fleet.get('loop_verification', {}).get('min_inliers', 30)), min_inlier_ratio=0.2,
            online_max_frames=8192, online_map_fps=1.0, online_max_points=3000, online_preview_fps=2.0,
            enable_dense_mapping=False, dense_fps=0.5, dense_max_points=50000,
            dense_voxel_size=0.02, dense_engine="/output/da3-small-238x378/da3.engine",
            online_check_health=True, loop_diagnostics_output=f'/output/{robot_id}-loops.json',
        )
        camera = dict(camera_type=robot.get('camera_type', 'realsense'), camera_serial=str(robot.get('camera_serial', '')),
                      camera_fps=profile['camera_fps'], camera_stride=profile['camera_stride'],
                      camera_format=profile['camera_format'], output_width=384, output_height=240)
        if camera['camera_type'] == 'zed':
            camera.update(camera_fps=15, camera_stride=2, camera_format='bgr8',
                          camera_device='/dev/dpvo_camera',
                          camera_calibration=f"/models/SN{camera['camera_serial']}.conf")
        for key in ('camera_fps', 'camera_stride'):
            if key in robot:
                camera[key] = int(robot[key])
            if camera[key] < 1:
                raise ValueError(f'{key} must be positive')
        write(f'{robot_id}.worker.yaml', {
            f'/{robot_id}/dpvo_multi_robot': {'ros__parameters': parameters},
            f'/{robot_id}/dpvo_camera': {'ros__parameters': camera},
        })
        write(f'{robot_id}.control.yaml', {'/**': {'ros__parameters': {
            'robot_id': robot_id, 'worker_parameters': f'/fleet/{robot_id}.worker.yaml'}}})
        (output / f'{robot_id}.json').write_text(json.dumps({"camera_serial": "", "camera_type": "realsense", **profile, **robot}, indent=2))
    write('coordinator.yaml', {
        '/cbs_pgo': {'ros__parameters': dict(
            robot_ids=ids, anchor_robot_id=fleet['coordinator']['anchor_robot'],
            output_dir='/output/cbs', cbs_executable='/opt/cbs/run',
            online_period=float(fleet['coordinator'].get('online_period', 10.0)),
            iterations=100, timeout_seconds=60.0,
            run_centralized_baseline=False, run_explicit_anchor_centralized_baseline=False)},
        '/dpvo_fleet_viewer': {'ros__parameters': {
            'robot_ids': ids, 'web_port': int(fleet['coordinator']['web_port'])}},
    })
    cyclone = ET.Element('CycloneDDS', xmlns='https://cdds.io/config')
    domain = ET.SubElement(cyclone, 'Domain', Id='any')
    general = ET.SubElement(domain, 'General')
    ET.SubElement(general, 'AllowMulticast').text = 'false'
    interfaces = ET.SubElement(general, 'Interfaces')
    direct = fleet.get('network', {}).get('transport', 'zenoh') == 'cyclone'
    # Direct DDS selects the active LAN interface at process startup.
    # Bridge mode keeps DDS local, including when Wi-Fi is absent.
    interface_address = fleet.get('network', {}).get('interface_address')
    selection = {'address': '127.0.0.1'}
    if direct:
        selection = ({'address': interface_address} if interface_address else
                     {'autodetermine': 'true'})
    ET.SubElement(interfaces, 'NetworkInterface', **selection)
    discovery = ET.SubElement(domain, 'Discovery')
    ET.SubElement(discovery, 'ParticipantIndex').text = 'auto'
    ET.SubElement(discovery, 'MaxAutoParticipantIndex').text = '64'
    peers = ET.SubElement(discovery, 'Peers')
    ET.SubElement(peers, 'Peer', Address='127.0.0.1')
    if direct:
        for address in sorted({r['address'] for r in fleet['robots']} | {fleet['coordinator']['address']}):
            ET.SubElement(peers, 'Peer', Address=address)
    hosts = {r['id']: r['address'] for r in fleet['robots']}
    if fleet['coordinator']['address'] not in hosts.values():
        hosts['coordinator'] = fleet['coordinator']['address']
    port = int(fleet.get('network', {}).get('bridge_port', 17473))
    topic_pattern = '/(' + '|'.join(re.escape(r) for r in ids) + ')/dpvo/.*'
    for host, address in hosts.items():
        bridge = {
            'mode': 'peer',
            'listen': {'endpoints': [f'tcp/0.0.0.0:{port}']},
            'connect': {'endpoints': [f'tcp/{ip}:{port}' for ip in hosts.values() if ip != address],
                        'timeout_ms': -1, 'exit_on_failure': False,
                        'retry': {'period_init_ms': 1000, 'period_max_ms': 4000,
                                  'period_increase_factor': 2}},
            'scouting': {'multicast': {'enabled': False}, 'gossip': {'enabled': False}},
            'plugins': {'ros2dds': {
                'domain': int(fleet['ros_domain_id']),
                'nodename': f'dpvo_network_{host}',
                'ros_automatic_discovery_range': 'SYSTEM_DEFAULT',
                'allow': {kind: [topic_pattern, '/dpvo_multi_robot/.*'] for kind in
                          ('publishers', 'subscribers', 'service_servers', 'service_clients')},
                # Camera/worker progress must never block on Wi-Fi congestion.
                'reliable_routes_blocking': False,
            }},
        }
        (output / f'{host}.bridge.json').write_text(json.dumps(bridge, indent=2))
    ET.indent(cyclone)
    (output / 'cyclonedds.xml').write_text(ET.tostring(cyclone, encoding='unicode'))
    (output / 'fleet.json').write_text(json.dumps(fleet, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fleet', type=Path, default=PROFILES / 'fleet.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(generate(load_fleet(args.fleet), args.output))


if __name__ == '__main__':
    main()

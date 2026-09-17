"""CPU regression checks for live fleet configuration and process/session isolation."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2/dpvo_multi_robot'))
from dpvo_multi_robot.online_common import WorkerProcess, connected_robots, parse_session_frame, session_frame

spec = importlib.util.spec_from_file_location('fleet', ROOT / 'deploy/jetson/multi_robot/fleet.py')
fleet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet)


class FleetConfigTest(unittest.TestCase):
    def test_loop_threshold_trial_and_rollback_apply_to_every_robot(self):
        config = yaml.safe_load((fleet.PROFILES / 'fleet.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            for threshold in (15, 30):
                config['loop_verification'] = {'min_inliers': threshold}
                fleet.generate(config, directory)
                for robot in config['robots']:
                    rid = robot['id']
                    worker = yaml.safe_load((Path(directory) / f'{rid}.worker.yaml').read_text())
                    params = worker[f'/{rid}/dpvo_multi_robot']['ros__parameters']
                    self.assertEqual(params['min_inliers'], threshold)
                    self.assertEqual(params['min_inlier_ratio'], 0.2)

    def test_coordinator_can_share_robot_address_without_duplicate_dds_peers(self):
        import xml.etree.ElementTree as ET
        config = yaml.safe_load((fleet.PROFILES / 'fleet.yaml').read_text())
        config['network']['transport'] = 'zenoh'
        config['coordinator']['address'] = config['robots'][0]['address']
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'fleet.yaml'
            source.write_text(yaml.safe_dump(config))
            fleet.generate(fleet.load_fleet(source), Path(directory) / 'generated')
            root = ET.parse(Path(directory) / 'generated/cyclonedds.xml')
            peers = root.findall('.//{https://cdds.io/config}Peer')
            self.assertEqual([p.attrib['Address'] for p in peers], ['127.0.0.1'])
            import json
            bridge = json.loads((Path(directory) / 'generated/robot0.bridge.json').read_text())
            self.assertEqual(len(bridge['connect']['endpoints']), len(config['robots']) - 1)
            self.assertNotIn('tcp/' + config['robots'][0]['address'] + ':17473', bridge['connect']['endpoints'])
            self.assertFalse(bridge['connect']['exit_on_failure'])
            interfaces = root.findall('.//{https://cdds.io/config}NetworkInterface')
            self.assertEqual([i.attrib['address'] for i in interfaces], ['127.0.0.1'])

    def test_direct_dds_uses_lan_peers_without_duplicate_coordinator(self):
        import xml.etree.ElementTree as ET
        config = yaml.safe_load((fleet.PROFILES / 'fleet.yaml').read_text())
        config['network']['transport'] = 'cyclone'
        with tempfile.TemporaryDirectory() as directory:
            fleet.generate(config, Path(directory))
            root = ET.parse(Path(directory) / 'cyclonedds.xml')
            peers = [p.attrib['Address'] for p in root.findall('.//{https://cdds.io/config}Peer')]
            self.assertEqual(set(peers), {'127.0.0.1'} | {r['address'] for r in config['robots']})
            self.assertEqual(len(peers), len(set(peers)))
            interface = root.find('.//{https://cdds.io/config}NetworkInterface')
            self.assertEqual(interface.attrib, {'address': '192.168.0.0'})
            config['network'].pop('interface_address')
            fleet.generate(config, Path(directory))
            root = ET.parse(Path(directory) / 'cyclonedds.xml')
            self.assertEqual(root.find('.//{https://cdds.io/config}NetworkInterface').attrib,
                             {'autodetermine': 'true'})

    def test_bridge_port_must_not_conflict_with_panel(self):
        config = yaml.safe_load((fleet.PROFILES / 'fleet.yaml').read_text())
        for port in (80, 9090, config['coordinator']['web_port'], 65536):
            config['network'] = {'bridge_port': port}
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'fleet.yaml'
                source.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    fleet.load_fleet(source)

    def test_two_three_four_robot_configs(self):
        original = yaml.safe_load((fleet.PROFILES / 'fleet.yaml').read_text())
        for count in (2, 3, 4):
            config = copy.deepcopy(original)
            config['robots'] = config['robots'][:2]
            for index in range(2, count):
                config['robots'].append(dict(id=f'robot{index}', address=f'192.168.0.{190+index}',
                                             profile='jetpack62', camera_serial=''))
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'fleet.yaml'
                source.write_text(yaml.safe_dump(config))
                fleet.generate(fleet.load_fleet(source), Path(directory) / 'generated')
                generated = Path(directory) / 'generated'
                coordinator_text = (generated / 'coordinator.yaml').read_text()
                self.assertNotIn('&id', coordinator_text)
                self.assertNotIn('*id', coordinator_text)
                coordinator = yaml.safe_load(coordinator_text)
                self.assertEqual(len(coordinator['/cbs_pgo']['ros__parameters']['robot_ids']), count)
                for robot in config['robots']:
                    rid = robot['id']
                    worker = yaml.safe_load((generated / f'{rid}.worker.yaml').read_text())
                    params = worker[f'/{rid}/dpvo_multi_robot']['ros__parameters']
                    self.assertTrue(params['session_frame_ids'])
                    self.assertEqual(params['image_topic'], f'/{rid}/camera/image_rect')
                    camera = worker[f'/{rid}/dpvo_camera']['ros__parameters']
                    self.assertEqual((camera['output_width'], camera['output_height']), (384, 240))
                self.assertIn('AllowMulticast>false', (generated / 'cyclonedds.xml').read_text())

    def test_rejects_ambiguous_fleet(self):
        original = yaml.safe_load((fleet.PROFILES / 'fleet.yaml').read_text())
        for mutate in (
            lambda c: c['robots'][1].update(id='robot0'),
            lambda c: c['robots'][1].update(address=c['robots'][0]['address']),
            lambda c: c['robots'][1].update(profile='unknown'),
            lambda c: c['coordinator'].update(online_period=-1),
        ):
            config = copy.deepcopy(original)
            mutate(config)
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'fleet.yaml'
                source.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    fleet.load_fleet(source)

    def test_connectivity_requires_every_robot(self):
        self.assertFalse(connected_robots(['r0', 'r1', 'r2', 'r3'], [('r0', 'r1'), ('r2', 'r3')]))
        self.assertTrue(connected_robots(['r0', 'r1', 'r2', 'r3'], [('r0', 'r1'), ('r1', 'r2'), ('r2', 'r3')]))
        self.assertFalse(connected_robots(['r0', 'r1'], [('other', 'r0')]))

    def test_session_roundtrip_and_invalid_frames(self):
        self.assertEqual(parse_session_frame(session_frame('robot0', '012345-abcdef')), ('robot0', '012345-abcdef'))
        for frame in ('map', 'robot0//map', '../bad/map', 'robot0/session/odom'):
            with self.assertRaises(ValueError):
                parse_session_frame(frame)


class WorkerLifecycleTest(unittest.TestCase):
    def test_start_stop_start_has_new_pid_and_fresh_memory(self):
        worker = WorkerProcess([sys.executable, '-c', 'import time; time.sleep(60)'], stop_timeout=.5)
        try:
            self.assertFalse(worker.running)
            self.assertTrue(worker.start())
            first = worker.process.pid
            self.assertFalse(worker.start())
            worker.stop()
            self.assertFalse(worker.running)
            with self.assertRaises(ProcessLookupError):
                os.kill(first, 0)
            self.assertTrue(worker.start())
            self.assertNotEqual(worker.process.pid, first)
        finally:
            worker.stop()

    def test_failed_parent_descendant_is_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            child_file = Path(directory) / 'child'
            code = ('import subprocess,sys,pathlib; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                    f'pathlib.Path({str(child_file)!r}).write_text(str(p.pid))')
            worker = WorkerProcess([sys.executable, '-c', code], stop_timeout=.5)
            try:
                worker.start()
                worker.process.wait(timeout=5)
                child = int(child_file.read_text())
                self.assertFalse(worker.running)
                for _ in range(100):
                    stat = Path(f'/proc/{child}/stat')
                    if not stat.exists() or stat.read_text().split()[2] == 'Z':
                        break
                    time.sleep(.01)
                else:
                    self.fail('Descendant survived worker exit')
            finally:
                worker.stop()


if __name__ == '__main__':
    unittest.main()

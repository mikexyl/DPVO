"""Docker network fault test: offline boot, attachment, loss, reconnect.

Uses only synthetic messages in isolated network namespaces and ROS domain 177.
Pass --binary with the extracted bridge executable for this machine.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import yaml
from fleet import generate


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--image', default='dpvo:online-coordinator-partial')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    prefix = 'dpvo-net-' + uuid.uuid4().hex[:8]
    names = [prefix + '-0', prefix + '-1']
    created = []
    with tempfile.TemporaryDirectory(prefix='dpvo-net-') as temporary:
        fleet = yaml.safe_load((here / 'fleet.yaml').read_text())
        fleet['ros_domain_id'] = 177
        fleet['robots'] = fleet['robots'][:2]
        for i, robot in enumerate(fleet['robots']):
            robot['address'] = f'172.30.174.{10+i}'
        fleet['coordinator']['address'] = fleet['robots'][0]['address']
        generate(fleet, Path(temporary))
        def stats():
            return [json.loads(run('docker', 'exec', n, 'cat', '/tmp/probe.json')) for n in names]
        def wait_for(predicate, label, timeout=40):
            until = time.monotonic() + timeout
            while time.monotonic() < until:
                try:
                    values = stats()
                    if predicate(values):
                        print(label, json.dumps(values), flush=True)
                        return values
                except (subprocess.CalledProcessError, json.JSONDecodeError):
                    pass
                time.sleep(.5)
            raise RuntimeError(f'{label}: timed out; stats={stats()}')
        try:
            run('docker', 'network', 'create', '--subnet', '172.30.174.0/24', prefix)
            for i, name in enumerate(names):
                run('docker', 'run', '-d', '--name', name, '--network', 'none',
                    '-e', 'ROS_DOMAIN_ID=177', '-e', 'CYCLONEDDS_URI=file:///fleet/cyclonedds.xml',
                    '-v', f'{temporary}:/fleet:ro', '-v', f'{args.binary.resolve()}:/bridge:ro',
                    '-v', f'{here / "network_probe.py"}:/probe.py:ro', args.image,
                    'python', '/probe.py', f'robot{i}', f'robot{1-i}', '/tmp/probe.json')
                created.append(name)
                run('docker', 'exec', '-d', name, 'sh', '-c',
                    f'/bridge -c /fleet/robot{i}.bridge.json > /tmp/bridge.log 2>&1')
            initial = wait_for(lambda s: all(x['local'] >= 5 and x['remote'] == 0 for x in s), 'OFFLINE BOOT')
            for i, name in enumerate(names):
                run('docker', 'network', 'disconnect', 'none', name)
                run('docker', 'network', 'connect', '--ip', f'172.30.174.{10+i}', prefix, name)
            connected = wait_for(lambda s: all(x['remote'] >= 5 and x['services'] >= 3 for x in s), 'CONNECTED')
            run('docker', 'network', 'disconnect', prefix, names[0])
            time.sleep(15)
            lost = stats()
            time.sleep(3)
            offline = stats()
            assert all(b['local'] > a['local'] and b['remote'] == a['remote'] for a, b in zip(lost, offline)), (lost, offline)
            print('DISCONNECTED: local messages continue; remote messages stopped', flush=True)
            run('docker', 'network', 'connect', '--ip', '172.30.174.10', prefix, names[0])
            recovered = wait_for(lambda s: all(x['remote'] > old['remote'] + 5 and x['services'] > old['services'] + 2 for x, old in zip(s, offline)), 'RECONNECTED')
            assert [s['boot'] for s in recovered] == [s['boot'] for s in initial]
            print('PASS: unchanged ROS processes across offline boot, connection, loss and reconnection', flush=True)
        finally:
            for name in created:
                subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL)
            subprocess.run(['docker', 'network', 'rm', prefix], stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()

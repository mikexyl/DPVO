"""One local native CBS optimizer, with peer-to-peer ROS belief exchange."""
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
import yaml


class NativeAgent:
    def __init__(self, executable, graph, robot, cancelled, timeout=30, parameters=None):
        self.directory = tempfile.TemporaryDirectory(prefix='dpvo-cbs-agent-')
        path = Path(self.directory.name) / 'graph.json'
        path.write_text(json.dumps(graph))
        self.lines = queue.Queue()
        self.cancelled, self.timeout = cancelled, timeout
        flags = [f'--{key}={value}' for key, value in (parameters or {}).items()
                 if key in ('target_hellinger', 'contract_alpha', 'd_reset', 'cbs_relinearize_threshold')]
        self.process = subprocess.Popen([executable, f'--input_graph={path}',
            f'--output_dir={self.directory.name}', f'--agent_robot={robot}',
            '--run_centralized=false', '--run_explicit_anchor_centralized=false',
            '--write_rerun_rrd=false', '--rerun_stream=false'] + flags,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.ready = self.receive()
            if self.ready.get('optimizer_instances') != 1:
                raise RuntimeError('Native process must own exactly one optimizer')
        except Exception:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            if line.startswith('CBS_RPC '):
                self.lines.put(line[8:])
        self.lines.put(None)

    def receive(self):
        end = time.monotonic() + self.timeout
        while time.monotonic() < end and not self.cancelled.is_set():
            try:
                line = self.lines.get(timeout=.2)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f'Native CBS process exited ({self.process.poll()})')
            result = yaml.safe_load(line)
            if not result.get('ok'):
                raise RuntimeError(result.get('error', 'Native CBS failed'))
            return result
        raise RuntimeError('Native CBS operation cancelled or timed out')

    def call(self, **request):
        self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=2)
        self.directory.cleanup()


class CbsAgent(Node):
    def __init__(self, **options):
        super().__init__('dpvo_cbs_agent', **options)
        self.declare_parameter('robot_id', 'robot0')
        self.declare_parameter('agent_executable', '/opt/cbs/agent')
        self.robot = self.get_parameter('robot_id').value
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.worker = None
        self.epoch = None
        self.job = None
        self.released = False
        self.inbox, self.outgoing = {}, {}
        self.boot = uuid.uuid4().hex
        self.state = dict(phase='idle', round=0, sent=0, received=0)
        self.status_pub = self.create_publisher(String, '/dpvo_multi_robot/cbs/agent_status', 50)
        self.belief_pub = self.create_publisher(String, '/dpvo_multi_robot/cbs/beliefs', 256)
        self.result_pub = self.create_publisher(String, '/dpvo_multi_robot/cbs/agent_results', 50)
        self.create_subscription(String, '/dpvo_multi_robot/cbs/jobs', self.on_job,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/dpvo_multi_robot/cbs/release', self.on_release, 20)
        self.create_subscription(String, '/dpvo_multi_robot/cbs/beliefs', self.on_belief, 256)
        self.create_timer(.5, self.report)

    def report(self):
        with self.lock:
            value = dict(robot=self.robot, epoch=self.epoch, boot=self.boot,
                         host=os.uname().nodename, pid=os.getpid(), **self.state)
        self.status_pub.publish(String(data=json.dumps(value)))

    def on_release(self, message):
        try:
            value = json.loads(message.data)
            with self.lock:
                if value.get('epoch') == self.epoch:
                    if value.get('cancel'):
                        self.cancel.set()
                    else:
                        self.released = True
        except (ValueError, TypeError):
            return

    def on_job(self, message):
        try:
            job = json.loads(message.data)
            if self.robot not in job['sessions'] or not 2 <= len(job['sessions']) <= 4:
                return
            if not 1 <= job['iterations'] <= 200 or time.time() > job['expires']:
                return
        except (ValueError, KeyError, TypeError):
            return
        with self.lock:
            if job['epoch'] == self.epoch:
                return
            if self.worker is not None and self.worker.is_alive():
                # The scheduler never overlaps jobs involving this robot.
                return
            self.epoch, self.job = job['epoch'], job
            self.cancel = threading.Event()
            self.inbox, self.outgoing = {}, {}
            self.released = False
            self.state = dict(phase='initializing', round=0, sent=0, received=0)
            self.worker = threading.Thread(target=self.run_job, args=(job,), daemon=True)
            self.worker.start()

    def on_belief(self, message):
        try:
            packet = json.loads(message.data)
            if packet.get('epoch') != self.epoch or packet.get('sender') == self.robot:
                return
            if packet.get('sender') not in (self.job or {}).get('sessions', {}):
                return
            round_id = packet['round']
            if not isinstance(round_id, int) or not 0 <= round_id < self.job['iterations']:
                return
            with self.lock:
                if packet.get('need') == self.robot:
                    stored = self.outgoing.get(round_id)
                    if stored is not None:
                        self.belief_pub.publish(String(data=json.dumps(stored)))
                elif 'beliefs' in packet and packet.get('need') is None:
                    self.inbox.setdefault(round_id, {}).setdefault(packet['sender'], packet)
        except (ValueError, KeyError, TypeError):
            return

    def _check(self, deadline):
        if self.cancel.is_set() or time.monotonic() > deadline:
            raise RuntimeError('Peer exchange cancelled or timed out')

    def run_job(self, job):
        native = None
        try:
            native = NativeAgent(self.get_parameter('agent_executable').value,
                                 job['graph'], self.robot, self.cancel, parameters=job.get('parameters'))
            neighbors = sorted(native.ready['neighbors'])
            with self.lock:
                self.state.update(phase='ready', optimizer_instances=1, native_pid=native.process.pid,
                                  neighbors=neighbors)
            self.report()
            deadline = time.monotonic() + 45
            while not self.released:
                self._check(deadline)
                self.cancel.wait(.05)
            for round_id in range(job['iterations']):
                self._check(time.monotonic() + 1)
                warmup = job.get('warmup', 0)
                pose_block, anchor_block = job.get('pose_block', 20), job.get('anchor_block', 20)
                anchor = round_id >= warmup and (round_id-warmup) % (pose_block+anchor_block) >= pose_block
                output = native.call(op='snapshot', anchor=anchor)
                packet = dict(epoch=job['epoch'], sender=self.robot, round=round_id,
                              anchor=anchor, beliefs=output['beliefs'])
                with self.lock:
                    self.outgoing[round_id] = packet
                    self.state.update(phase='optimizing', round=round_id + 1)
                    self.state['sent'] += len(neighbors)
                self.belief_pub.publish(String(data=json.dumps(packet, allow_nan=False)))
                deadline, retry = time.monotonic() + 20, 0.
                while True:
                    self._check(deadline)
                    with self.lock:
                        received = self.inbox.get(round_id, {})
                        missing = [r for r in neighbors if r not in received]
                        if not missing:
                            packets = [received[r] for r in neighbors]
                            break
                    if time.monotonic() > retry:
                        for peer in missing:
                            self.belief_pub.publish(String(data=json.dumps(dict(
                                epoch=job['epoch'], sender=self.robot, round=round_id, need=peer))))
                        retry = time.monotonic() + .5
                    self.cancel.wait(.02)
                native.call(op='update', packets=packets)
                with self.lock:
                    self.state['received'] += len(packets)
                    self.inbox.pop(round_id, None)
            result = native.call(op='result')
            result.update(epoch=job['epoch'], robot=self.robot, sessions=job['sessions'],
                          sent=self.state['sent'], received=self.state['received'],
                          optimizer_instances=1, host=os.uname().nodename)
            output = Path('/output')
            if output.is_dir() and os.access(output, os.W_OK):
                (output / 'latest.json').write_text(json.dumps(result, indent=2))
            self.result_pub.publish(String(data=json.dumps(result, allow_nan=False)))
            with self.lock:
                self.state['phase'] = 'complete'
        except Exception as error:
            with self.lock:
                self.state.update(phase='error', error=str(error))
            self.get_logger().error(str(error))
            self.result_pub.publish(String(data=json.dumps(dict(
                epoch=job['epoch'], robot=self.robot, ok=False, error=str(error)))))
        finally:
            if native is not None:
                native.close()
            self.report()

    def close(self):
        self.cancel.set()
        if self.worker is not None:
            self.worker.join(timeout=35)


def main():
    rclpy.init()
    node = CbsAgent()
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

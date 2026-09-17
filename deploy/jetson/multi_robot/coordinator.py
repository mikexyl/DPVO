"""Supervise both CPU services; any failure restarts the container as a unit."""
import os
import signal
import sys
import time
from dpvo_multi_robot.online_common import WorkerProcess


def main():
    workers = [WorkerProcess([sys.executable, '-m', module, '--ros-args',
               '--params-file', '/fleet/coordinator.yaml']) for module in
               (os.environ.get('DPVO_CBS_MODULE', 'dpvo_multi_robot.distributed_cbs'), 'dpvo_multi_robot.online_viewer')]
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for worker in workers:
            worker.start()
        while not stopping:
            if not all(worker.running for worker in workers):
                raise RuntimeError('A coordinator service exited; restarting both services')
            time.sleep(.2)
    finally:
        for worker in workers:
            worker.stop()


if __name__ == '__main__':
    main()

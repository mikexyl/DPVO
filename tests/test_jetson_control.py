"""Control-panel worker lifecycle regressions (no camera or CUDA required)."""
import multiprocessing as mp
from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy' / 'jetson'))
from viser_control import WorkerViewer, stop_worker


def idle_worker(ready):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    ready.set()
    while True:
        time.sleep(0.1)


class ControlTests(unittest.TestCase):
    def test_importing_web_entrypoint_does_not_initialize_cuda(self):
        directory = Path(__file__).resolve().parents[1] / 'deploy' / 'jetson'
        subprocess.run(
            [sys.executable, '-c',
             'import live_realsense; import torch; '
             'assert not torch.cuda.is_initialized(), "Idle UI initialized CUDA"'],
            cwd=directory, check=True, timeout=30)

    def test_worker_is_reaped_before_stop_returns(self):
        context = mp.get_context('spawn')
        ready = context.Event()
        worker = context.Process(target=idle_worker, args=(ready,))
        worker.start()
        self.assertTrue(ready.wait(15))
        pid = worker.pid
        stop_worker(worker)
        self.assertNotIn(pid, [child.pid for child in mp.active_children()])
        with self.assertRaises(ValueError):
            worker.is_alive()  # Process handle itself was closed as well.
        stop_worker(None)

    def test_dense_snapshot_converts_stable_gauge_and_resets(self):
        queue = Queue(maxsize=2)
        gauge = np.eye(4)
        gauge[:3, :3] *= 2
        gauge[:3, 3] = [1, 2, 3]
        graph = SimpleNamespace(poses_=torch.zeros(0, 7), points_=torch.zeros(0, 3),
                                colors_=torch.zeros(0, 3), session_from_map_=gauge)
        viewer = WorkerViewer(queue, graph)
        viewer.update_dense(np.array([[3., 6., 9.]]), np.array([[10, 20, 30]], np.uint8))
        viewer.update_state(torch.ones(4), 0, 0)
        frame = queue.get_nowait()
        np.testing.assert_allclose(frame['dense_points'], [[1., 2., 3.]])
        np.testing.assert_array_equal(frame['dense_colors'], [[10, 20, 30]])
        viewer.reset()
        viewer.update_state(torch.ones(4), 0, 0)
        self.assertNotIn('dense_points', queue.get_nowait())

    def test_snapshots_are_cpu_bounded_and_do_not_block(self):
        queue = Queue(maxsize=1)
        graph = SimpleNamespace(poses_=torch.zeros(2, 7),
                                points_=torch.ones(20000, 3),
                                colors_=torch.zeros(20000, 3, dtype=torch.uint8))
        viewer = WorkerViewer(queue, graph)
        viewer.update_image(torch.zeros(3, 240, 384, dtype=torch.uint8))
        viewer.update_state(torch.ones(4), 2, 20000)
        viewer.update_state(torch.ones(4), 2, 20000)
        frame = queue.get_nowait()
        self.assertLessEqual(len(frame['points']), 6000)
        self.assertEqual(frame['image'].shape, (240, 384, 3))
        viewer.reset()
        viewer.update_state(torch.ones(4), 0, 0)
        self.assertEqual(queue.get_nowait()['generation'], frame['generation'] + 1)


if __name__ == '__main__':
    unittest.main()

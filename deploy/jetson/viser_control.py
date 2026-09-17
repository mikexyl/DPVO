"""Persistent web panel with an expendable camera/CUDA worker per run."""
import json
import os
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import signal
import time
from types import SimpleNamespace

import numpy as np


class WorkerViewer:
    """Send bounded CPU scene snapshots; never share CUDA resources with the UI."""

    def __init__(self, messages, patch_graph, *args, **kwargs):
        self.messages = messages
        self.patch_graph = patch_graph
        self.num_frames = self.num_points = 0
        self.generation = 0
        self.image = None
        self.status = ''
        self.dense = None

    def set_status(self, text):
        self.status = text

    def set_running(self, running):
        pass

    def take_command(self):
        return None

    def reset(self):
        self.generation += 1
        self.dense = None

    def update_dense(self, points, colors):
        self.dense = (points.copy(), colors.copy())

    def update_image(self, image):
        self.image = image.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1].copy()

    def update_state(self, intrinsics, num_frames, num_points):
        step = max(1, (num_points + 5999) // 6000)
        graph = self.patch_graph
        snapshot = dict(
            generation=self.generation, image=self.image, status=self.status,
            intrinsics=intrinsics.detach().cpu().numpy(),
            poses=graph.poses_[:num_frames].detach().cpu().numpy(),
            points=graph.points_[:num_points:step].detach().cpu().numpy(),
            colors=graph.colors_.reshape(-1, 3)[:num_points:step].detach().cpu().numpy())
        if self.dense is not None:
            # Standalone sparse display uses the current DPVO map gauge; the
            # shared dense mapper emits stable-session coordinates.
            inverse = np.linalg.inv(graph.session_from_map_)
            points, colors = self.dense
            snapshot['dense_points'] = (points @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.float32)
            snapshot['dense_colors'] = colors
        try:
            self.messages.put_nowait(snapshot)
        except Full:
            pass  # A slow browser must never stall capture or inference.

    def join(self):
        pass


def run_worker(args, messages):
    os.setsid()  # Own the tracker and its spawned TensorRT child as one group.
    from live_realsense import main
    # Do not wait for unconsumed snapshots during worker shutdown.
    messages.cancel_join_thread()
    args.start_paused = False
    main(args, viewer_factory=lambda *a, **kw: WorkerViewer(messages, *a, **kw))


def stop_worker(worker):
    if worker is None:
        return
    try:
        group = worker.pid if getattr(worker, 'dpvo_group', False) or (worker.pid and os.getpgid(worker.pid) == worker.pid) else None
    except ProcessLookupError:
        group = None
    if worker.is_alive():
        if group is not None:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                worker.terminate()
        else:
            worker.terminate()  # SIGTERM permits pipeline.stop() in the runner.
        worker.join(timeout=8)
    if worker.is_alive():
        worker.kill()
    if group is not None:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    worker.join()
    worker.close()


def run_control_panel(args):
    import torch
    from dpvo.viser_viewer import ViserViewer

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    viewer = ViserViewer(None, 240, 384, web_port=args.web_port, dense_controls=True)
    context = mp.get_context('spawn')
    worker = messages = None
    stopping = False
    session = 0
    generation = None

    def shutdown(*_):
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, shutdown)

    def write_status(state, error=None):
        status = dict(state=state, tracking_enabled=state == 'starting',
                      camera_running=False, worker_pid=worker.pid if worker else None,
                      viewer='viser', session=session, processed_frames=0,
                      captured_frames=0, processing_fps=0., keyframes=0, points=0)
        if error:
            status['error'] = error
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(output / 'status.json')

    def clear_scene():
        viewer.patch_graph = None
        viewer.image = None
        viewer.reset()
        viewer.preview.image = np.zeros((240, 384, 3), dtype=np.uint8)

    def stop_session(error=None):
        nonlocal worker, messages
        viewer.set_status('Stopping camera and DPVO…')
        stop_worker(worker)
        worker = None
        if messages is not None:
            messages.close()
            messages = None
        clear_scene()
        write_status('error' if error else 'paused', error)
        viewer.set_status(f'Start failed or tracking stopped: {error}. Press Start DPVO to retry.'
                          if error else 'Camera and DPVO are off. Press Start DPVO to begin a fresh map.')
        viewer.set_running(False)

    stop_session()
    auto_start = not args.start_paused
    try:
        while not stopping:
            command = True if auto_start else viewer.take_command()
            auto_start = False
            if command is False:
                stop_session()
            elif command is True and worker is None:
                args.enable_dense_mapping = bool(viewer.dense_enabled.value)
                engine = Path(args.dense_engine)
                if args.enable_dense_mapping and not (engine.is_file() and engine.with_name('manifest.json').is_file()):
                    viewer.set_status('DA3 TensorRT engine is missing; disable dense mapping or prepare the engine.')
                    viewer.set_running(False)
                    continue
                session += 1
                generation = None
                clear_scene()
                messages = context.Queue(maxsize=2)
                worker = context.Process(target=run_worker, args=(args, messages))
                viewer.set_status('Starting camera and loading a fresh DPVO model…')
                worker.start()
                worker.dpvo_group = True
                write_status('starting')
                viewer.set_running(True)
            if worker is not None:
                if not worker.is_alive():
                    exitcode = worker.exitcode
                    stop_session(None if exitcode == 0 else f'worker exited ({exitcode}); check service logs')
                    continue
                try:
                    snapshot = messages.get_nowait()
                except Empty:
                    snapshot = None
                if snapshot is not None:
                    if generation != snapshot['generation']:
                        viewer.reset()
                        generation = snapshot['generation']
                    viewer.patch_graph = SimpleNamespace(
                        poses_=torch.from_numpy(snapshot['poses']),
                        points_=torch.from_numpy(snapshot['points']),
                        colors_=torch.from_numpy(snapshot['colors']))
                    viewer.image = snapshot['image']
                    viewer.update_state(snapshot['intrinsics'], len(snapshot['poses']),
                                        len(snapshot['points']))
                    if 'dense_points' in snapshot:
                        viewer.update_dense(snapshot['dense_points'], snapshot['dense_colors'])
                    viewer.set_status(snapshot['status'])
            time.sleep(0.03)
    finally:
        stop_session()
        viewer.join()

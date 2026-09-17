"""Bounded live DPVO scene for Viser's browser client."""
import numpy as np
from queue import Empty, SimpleQueue
import viser

def _invert_poses(poses):
    """Convert DPVO world-to-camera poses into camera-to-world poses."""
    poses = poses.detach().cpu().numpy()
    translation = poses[:, :3]
    quaternion = poses[:, 3:]

    inverse_quaternion = quaternion.copy()
    inverse_quaternion[:, :3] *= -1

    xyz = inverse_quaternion[:, :3]
    w = inverse_quaternion[:, 3:]
    uv = 2.0 * np.cross(xyz, translation)
    inverse_translation = -(translation + w * uv + np.cross(xyz, uv))

    return np.concatenate([inverse_translation, inverse_quaternion], axis=1)


class ViserViewer:
    def __init__(self, patch_graph, height, width, web_port=9090, max_points=6000, dense_controls=False):
        self.patch_graph = patch_graph
        self.height, self.width = height, width
        self.max_points = max_points
        self.num_frames = self.num_points = 0
        self.image = None
        self.closed = False
        self.server = viser.ViserServer(host='0.0.0.0', port=web_port, label='DPVO Live')
        self.server.scene.set_up_direction('-y')
        self.server.gui.configure_theme(control_layout='collapsible', control_width='small',
                                        show_share_button=False)
        self.status = self.server.gui.add_markdown('Starting camera…')
        self.commands = SimpleQueue()
        self.start_button = self.server.gui.add_button('Start DPVO', disabled=True)
        self.stop_button = self.server.gui.add_button('Stop DPVO')
        self.start_button.on_click(lambda _: self.request_tracking(True))
        self.stop_button.on_click(lambda _: self.request_tracking(False))
        self.dense_enabled = (self.server.gui.add_checkbox('Dense mapping (TensorRT)', initial_value=False)
                              if dense_controls else None)
        with self.server.gui.add_folder('Map display'):
            self.point_size = self.server.gui.add_slider(
                'Point size', min=0.001, max=0.05, step=0.001, initial_value=0.005)
            self.dense_size = self.server.gui.add_slider(
                'Dense point size', min=.001, max=.05, step=.001, initial_value=.005)
        self.preview = self.server.gui.add_image(
            np.zeros((height, width, 3), dtype=np.uint8), label='Live camera',
            format='jpeg', jpeg_quality=75)
        self.server.gui.add_markdown('Drag to orbit; pinch to zoom. Map scale is monocular.')
        self.center = np.zeros(3)
        self.distance = 3.0
        self.camera = self.server.scene.add_camera_frustum(
            '/camera', fov=1.0, aspect=width/height, scale=0.15, color=(0, 170, 255),
            visible=False)
        self.points = self.server.scene.add_point_cloud(
            '/points', points=np.zeros((1, 3), dtype=np.float32), colors=(255, 255, 255),
            point_size=self.point_size.value, precision='float32', visible=False)

        self.dense_points = self.server.scene.add_point_cloud(
            '/dense', points=np.zeros((1, 3), dtype=np.float32), colors=(255, 255, 255),
            point_size=self.dense_size.value, precision='float32', visible=False)
        @self.dense_size.on_update
        def on_dense_size(_):
            self.dense_points.point_size = self.dense_size.value

        @self.point_size.on_update
        def on_point_size(_):
            self.points.point_size = self.point_size.value

        self.trajectory = self.server.scene.add_line_segments(
            '/trajectory', points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(0, 170, 255), thickness=2, thickness_units='screen', visible=False)

        def recenter(client):
            client.camera.up_direction = (0, -1, 0)
            client.camera.position = self.center + self.distance * np.array([0.6, -0.5, -1])
            client.camera.look_at = self.center

        self._recenter = recenter
        self.server.on_client_connect(recenter)
        button = self.server.gui.add_button('Center view')

        @button.on_click
        def on_center(event):
            if event.client is not None:
                recenter(event.client)

    def set_status(self, text):
        self.status.content = text

    def request_tracking(self, running):
        # GUI callbacks run on Viser threads. Only the inference loop may
        # mutate DPVO or create CUDA state.
        self.start_button.disabled = self.stop_button.disabled = True
        if self.dense_enabled is not None:
            self.dense_enabled.disabled = True
        self.commands.put(running)

    def take_command(self):
        command = None
        while True:
            try:
                command = self.commands.get_nowait()
            except Empty:
                return command

    def set_running(self, running):
        self.start_button.disabled = running
        self.stop_button.disabled = not running
        if self.dense_enabled is not None:
            self.dense_enabled.disabled = running

    def update_preview(self, bgr):
        self.preview.image = bgr[:, :, ::-1].copy()

    def reset(self):
        self.num_frames = self.num_points = 0
        self.camera.visible = self.points.visible = self.trajectory.visible = False
        self.dense_points.visible = False
        self.dense_points.points = np.empty((0, 3), dtype=np.float32)
        self.center = np.zeros(3)
        self.distance = 3.0
        for client in self.server.get_clients().values():
            self._recenter(client)

    def update_image(self, image):
        # DPVO consumes OpenCV BGR; Viser expects RGB.
        self.image = image.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1].copy()

    def update_state(self, intrinsics, num_frames, num_points):
        self.num_frames, self.num_points = num_frames, num_points
        with self.server.atomic():
            if self.image is not None:
                self.preview.image = self.image
            if num_frames:
                poses = _invert_poses(self.patch_graph.poses_[:num_frames])
                self.camera.position = poses[-1, :3]
                self.camera.wxyz = poses[-1, [6, 3, 4, 5]]
                fy = float(intrinsics[1])
                self.camera.fov = 2 * np.arctan(self.height / (2 * fy))
                self.camera.visible = True
                self.center = np.median(poses[:, :3], axis=0)
                self.distance = max(2., float(np.linalg.norm(poses[:, :3]-self.center, axis=1).max())*1.5)
                selected = poses[np.linspace(0, num_frames-1, min(num_frames, 2048), dtype=int), :3]
                if len(selected) > 1:
                    self.trajectory.points = np.stack([selected[:-1], selected[1:]], axis=1)
                    self.trajectory.visible = True
            if num_points:
                step = max(1, (num_points + self.max_points - 1) // self.max_points)
                points = self.patch_graph.points_[:num_points:step].detach().cpu().numpy()
                colors = self.patch_graph.colors_.reshape(-1, 3)[:num_points:step].detach().cpu().numpy()
                valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-8)
                self.points.points = points[valid].astype(np.float32)
                self.points.colors = colors[valid]
                self.points.visible = bool(valid.any())

    def update_dense(self, points, colors):
        valid = np.isfinite(points).all(axis=1)
        self.dense_points.points = points[valid].astype(np.float32)
        self.dense_points.colors = colors[valid].astype(np.uint8)
        self.dense_points.visible = bool(valid.any())

    def join(self):
        if not self.closed:
            self.server.stop()
            self.closed = True

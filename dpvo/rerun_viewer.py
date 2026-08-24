from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


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


def _quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class RerunViewer:
    """Streams DPVO's camera, trajectory, and sparse map to Rerun."""

    def __init__(
        self,
        patch_graph,
        height,
        width,
        save_path=None,
        connect_url=None,
        recording_id=None,
        entity_prefix=None,
    ):
        self.patch_graph = patch_graph
        self.height = height
        self.width = width
        self.image = None
        self.intrinsics = None
        self.frame_index = 0
        self.num_frames = 0
        self.num_points = 0
        self.closed = False
        self.entity_prefix = (entity_prefix or "world").strip("/")
        self.apply_map_gauge = entity_prefix is not None and hasattr(
            patch_graph,
            "session_from_map_",
        )

        blueprint = None
        if entity_prefix is None:
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="Reconstruction", origin="world"),
                    rrb.Spatial2DView(name="Camera", origin="world/camera/image"),
                    column_shares=[2, 1],
                ),
                collapse_panels=True,
            )
        rr.init(
            "DPVO Multi Robot" if recording_id else "DPVO",
            recording_id=recording_id,
            default_blueprint=blueprint,
            strict=True,
        )

        if save_path is not None and connect_url is not None:
            raise ValueError("Rerun save path and connection URL are mutually exclusive")

        if connect_url is not None:
            rr.connect_grpc(connect_url)
        elif save_path is None:
            rr.spawn(memory_limit="50%")
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(save_path)

        rr.log(self.entity_prefix, rr.ViewCoordinates.RDF, static=True)

    def _entity(self, suffix=""):
        return (
            f"{self.entity_prefix}/{suffix.lstrip('/')}"
            if suffix
            else self.entity_prefix
        )

    def update_image(self, image):
        self.image = image.permute(1, 2, 0).detach().cpu().numpy()

    def update_state(self, intrinsics, num_frames, num_points):
        self.intrinsics = intrinsics.detach().cpu().numpy()
        self.num_frames = num_frames
        self.num_points = num_points
        self._log_state(self.frame_index)
        self.frame_index += 1

    def _log_state(self, frame_index):
        if self.image is None or self.intrinsics is None:
            return

        rr.set_time("frame", sequence=frame_index)
        if self.apply_map_gauge:
            gauge = self.patch_graph.session_from_map_
            rr.log(
                self.entity_prefix,
                rr.Transform3D(
                    translation=gauge[:3, 3],
                    mat3x3=gauge[:3, :3],
                ),
            )
        rr.log(
            self._entity("camera/image"),
            rr.Pinhole(
                focal_length=self.intrinsics[:2],
                principal_point=self.intrinsics[2:],
                resolution=[self.width, self.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.1,
            ),
        )
        rr.log(
            self._entity("camera/image/rgb"),
            rr.Image(self.image, color_model="BGR").compress(jpeg_quality=90),
        )

        if self.num_frames > 0:
            poses = _invert_poses(self.patch_graph.poses_[: self.num_frames])
            rr.log(
                self._entity("trajectory"),
                rr.LineStrips3D(
                    [poses[:, :3]],
                    colors=[0, 170, 255],
                    radii=rr.Radius.ui_points(2.0),
                ),
            )
            rr.log(
                self._entity("camera"),
                rr.Transform3D(
                    translation=poses[-1, :3],
                    mat3x3=_quaternion_to_matrix(poses[-1, 3:]),
                ),
            )

        if self.num_points > 0:
            points = self.patch_graph.points_[: self.num_points].detach().cpu().numpy()
            colors = (
                self.patch_graph.colors_.view(-1, 3)[: self.num_points]
                .detach()
                .cpu()
                .numpy()
            )
            valid = np.isfinite(points).all(axis=1) & (
                np.linalg.norm(points, axis=1) > 1e-8
            )
            rr.log(
                self._entity("points"),
                rr.Points3D(
                    points[valid],
                    colors=colors[valid],
                    radii=rr.Radius.ui_points(2.0),
                ),
            )

    def join(self):
        if self.closed:
            return

        if self.frame_index > 0:
            self._log_state(self.frame_index - 1)
        rr.disconnect()
        self.closed = True

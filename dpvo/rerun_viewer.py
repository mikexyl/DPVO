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
        yolo_model=None,
        yolo_confidence=0.25,
        yolo_image_size=640,
        yolo_task=None,
        scene_graph=False,
        scene_graph_output=None,
        da3_engine=None,
        dense_map_output=None,
        dense_map_stride=7,
        dense_map_max_error=0.25,
        keyframe_delay=4,
        sam_model=None,
        sam_points_per_side=32,
        sam_points_per_batch=16,
        sam_min_mask_area=100,
        sam_pred_iou_thresh=0.8,
        sam_stability_thresh=0.92,
        sam_output=None,
        sam_video=False,
        sam_video_max_tracks=8,
        sam_video_refresh=10,
        sam_video_memory=3,
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
        self.detections = None
        self.detector = None
        self.sam_segmenter = None
        self.sam_output = sam_output
        self.sam_video = sam_video
        self.scene_graph = None
        self.scene_graph_output = scene_graph_output
        self.dense_map = None
        self.dense_map_output = dense_map_output
        self.dense_updates = []
        self._dense_camera_to_world = None

        if sam_model is not None and yolo_model is not None:
            raise ValueError("SAM and YOLO segmentation backends are mutually exclusive")
        if sam_model is not None and scene_graph:
            raise ValueError("SAM's class-agnostic regions cannot populate the semantic scene graph")
        if sam_video and sam_model is None:
            raise ValueError("SAM video requires a SAM TensorRT bundle")

        if yolo_model is not None:
            from .yolo_detector import YoloTensorRTDetector

            self.detector = YoloTensorRTDetector(
                yolo_model,
                confidence=yolo_confidence,
                image_size=yolo_image_size,
                task=yolo_task,
            )

        if sam_model is not None:
            options = dict(points_per_side=sam_points_per_side, min_mask_area=sam_min_mask_area,
                           pred_iou_thresh=sam_pred_iou_thresh, stability_score_thresh=sam_stability_thresh)
            if sam_video:
                from .sam_video import Sam2VideoSegmenter
                self.sam_segmenter = Sam2VideoSegmenter(
                    sam_model, **options, max_tracks=sam_video_max_tracks,
                    refresh_interval=sam_video_refresh, memory_frames=sam_video_memory,
                )
            else:
                from .sam_segmenter import Sam2Segmenter
                self.sam_segmenter = Sam2Segmenter(sam_model, **options, points_per_batch=sam_points_per_batch)
            self.detector = self.sam_segmenter

        if scene_graph:
            from .scene_graph import SceneGraphBuilder

            self.scene_graph = SceneGraphBuilder()

        if da3_engine is not None:
            from .da3_dense import DenseMapBuilder, _camera_to_world

            self.dense_map = DenseMapBuilder(
                da3_engine,
                patch_graph,
                image_height=height,
                image_width=width,
                keyframe_delay=keyframe_delay,
                point_stride=dense_map_stride,
                max_alignment_error=dense_map_max_error,
                region_coloring=("sam_video_track_id" if sam_video else "sam_frame_local_id")
                if sam_model is not None else None,
            )
            self._dense_camera_to_world = _camera_to_world

        camera_view = rrb.Spatial2DView(
            name="Camera",
            origin="world/camera/image",
            # Dense region boxes obscure the masks; keep them available in the
            # recording, but hide them in the default SAM camera view.
            contents=["$origin/**", "- $origin/detections"] if sam_model else "$origin/**",
        )
        right_views = [camera_view]
        row_shares = [2]
        if self.scene_graph is not None:
            right_views.append(rrb.GraphView(name="Scene Graph", origin="scene_graph"))
            row_shares.append(1)
        if self.dense_map is not None:
            right_views.append(rrb.Spatial2DView(name="Aligned DA3 Depth", origin="dense"))
            row_shares.append(1)
        if len(right_views) == 1:
            right_panel = camera_view
        else:
            right_panel = rrb.Vertical(*right_views, row_shares=row_shares)
        reconstruction = rrb.Spatial3DView(name="Reconstruction", origin="world")
        if self.dense_map is not None and self.sam_segmenter is not None:
            reconstruction = rrb.Tabs(
                rrb.Spatial3DView(
                    name="SAM-colored Dense Map", origin="world",
                    contents=["$origin/**", "- $origin/dense/**", "- $origin/points"],
                ),
                rrb.Spatial3DView(
                    name="RGB Dense Map", origin="world",
                    contents=["$origin/**", "- $origin/dense_sam/**"],
                ),
                active_tab=0,
            )
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                reconstruction,
                right_panel,
                column_shares=[2, 1],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
        rr.init("DPVO", strict=True)

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

        # Send after attaching the sink so saved recordings include the layout.
        # Make it active to avoid reusing panels from a different DPVO experiment.
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        if self.sam_segmenter is not None:
            info = (
                "SAM 2.1 video: learned temporal mask memory, persistent track IDs/colors. "
                "TensorRT image encoder/seeding; BF16 PyTorch video attention, heads and memory encoder. "
                "Scores are object-presence estimates, not semantic classes. Periodic re-prompting "
                "refreshes memory and retains IDs only when current-frame masks can be associated."
                if sam_video else
                "SAM 2.1 Hiera Tiny: automatic per-frame regions, not semantic classes "
                "or tracked IDs. Scores are predicted mask IoU. "
                "Smaller regions take precedence over overlapping large surfaces."
            )
            rr.log("segmentation/info", rr.TextDocument(
                info
            ), static=True)

    def update_image(self, image):
        self.image = image.permute(1, 2, 0).detach().cpu().numpy()
        if self.dense_map is not None:
            self.dense_map.cache_image(self.frame_index, self.image)
        if self.detector is not None:
            self.detections = self.detector(self.image)
        if self.dense_map is not None and self.sam_segmenter is not None:
            self.dense_map.cache_regions(self.frame_index, self.detections)

    def update_state(self, intrinsics, num_frames, num_points):
        self.intrinsics = intrinsics.detach().cpu().numpy()
        self.num_frames = num_frames
        self.num_points = num_points
        if self.scene_graph is not None:
            self.scene_graph.update(
                self.detections,
                self.patch_graph,
                num_frames,
                self.frame_index,
            )
        if self.dense_map is not None:
            self.dense_updates = self.dense_map.update(num_frames)
        self._log_state(self.frame_index)
        self.frame_index += 1

    def _log_state(self, frame_index):
        if self.image is None or self.intrinsics is None:
            return

        rr.set_time("frame", sequence=frame_index)
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                focal_length=self.intrinsics[:2],
                principal_point=self.intrinsics[2:],
                resolution=[self.width, self.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.1,
            ),
        )
        rr.log(
            "world/camera/image/rgb",
            rr.Image(self.image, color_model="BGR").compress(jpeg_quality=90),
        )
        if self.detector is not None:
            detection_path = "world/camera/image/detections"
            if self.detections is None:
                rr.log(detection_path, rr.Clear(recursive=False))
            else:
                rr.log(
                    detection_path,
                    rr.Boxes2D(
                        array=self.detections.boxes,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=self.detections.class_ids,
                        labels=self.detections.labels,
                        colors=self.detections.colors,
                        show_labels=self.sam_segmenter is None,
                    ),
                )

            segmentation_path = "world/camera/image/segmentation"
            if self.detections is None or self.detections.segmentation is None:
                rr.log(segmentation_path, rr.Clear(recursive=True))
            else:
                rr.log(
                    segmentation_path,
                    rr.AnnotationContext(self.detections.segmentation_context),
                )
                rr.log(
                    f"{segmentation_path}/mask",
                    rr.SegmentationImage(
                        self.detections.segmentation,
                        opacity=0.5,
                    ),
                )
            if self.sam_segmenter is not None and self.sam_segmenter.last_stats is not None:
                for key in ("masks", "coverage", "seconds"):
                    rr.log(f"segmentation/{key}", rr.Scalars(self.sam_segmenter.last_stats[key]))
                if self.sam_video:
                    for key in ("active_tracks", "memory_records", "refresh"):
                        rr.log(f"segmentation/{key}", rr.Scalars(float(self.sam_segmenter.last_stats[key])))

        if self.num_frames > 0:
            poses = _invert_poses(self.patch_graph.poses_[: self.num_frames])
            rr.log(
                "world/trajectory",
                rr.LineStrips3D(
                    [poses[:, :3]],
                    colors=[0, 170, 255],
                    radii=rr.Radius.ui_points(2.0),
                ),
            )
            rr.log(
                "world/camera",
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
                "world/points",
                rr.Points3D(
                    points[valid],
                    colors=colors[valid],
                    radii=rr.Radius.ui_points(2.0),
                ),
            )

        if self.scene_graph is not None:
            self._log_scene_graph()
        if self.dense_map is not None:
            self._log_dense_map()

    def _log_dense_map(self):
        for frame in self.dense_updates:
            path = f"world/dense/keyframe_{frame.timestamp:06d}"
            rr.log(
                f"{path}/points",
                rr.Points3D(
                    frame.camera_points,
                    colors=frame.colors,
                    radii=rr.Radius.ui_points(1.0),
                ),
                static=True,
            )

            if frame.region_ids is not None:
                sam_path = f"world/dense_sam/keyframe_{frame.timestamp:06d}/points"
                rr.log(sam_path, rr.AnnotationContext(frame.region_context), static=True)
                rr.log(
                    sam_path,
                    rr.Points3D(frame.camera_points, colors=frame.region_colors,
                                class_ids=frame.region_ids, show_labels=False,
                                radii=rr.Radius.ui_points(1.0)),
                    static=True,
                )

        # Refresh recently added keyframe transforms while they are still inside
        # DPVO's optimization window. The point data itself remains static.
        recent = list(self.dense_map.frames.values())[-12:]
        for frame in recent:
            pose = self.dense_map.pose(frame.timestamp, self.num_frames)
            if pose is None:
                continue
            rotation, translation = self._dense_camera_to_world(pose)
            rr.log(
                f"world/dense/keyframe_{frame.timestamp:06d}",
                rr.Transform3D(translation=translation, mat3x3=rotation),
            )
            if frame.region_ids is not None:
                rr.log(f"world/dense_sam/keyframe_{frame.timestamp:06d}",
                       rr.Transform3D(translation=translation, mat3x3=rotation))

        if self.dense_updates and self.dense_updates[-1].aligned_depth is not None:
            frame = self.dense_updates[-1]
            finite_depth = frame.aligned_depth[np.isfinite(frame.aligned_depth)]
            if len(finite_depth):
                depth_range = np.quantile(finite_depth, [0.01, 0.99])
                rr.log(
                    "dense/depth",
                    rr.DepthImage(
                        frame.aligned_depth,
                        meter=1.0,
                        colormap="turbo",
                        depth_range=depth_range,
                    ),
                )
        # Depth images are only needed for the current Rerun update. Keeping every
        # full-resolution map would make long sequences consume unbounded memory.
        for frame in self.dense_updates:
            frame.aligned_depth = None
        self.dense_updates = []

    def _log_scene_graph(self):
        nodes = self.scene_graph.visible_nodes()
        if not nodes:
            rr.log("scene_graph", rr.Clear(recursive=True))
            rr.log("world/scene_graph", rr.Clear(recursive=True))
            return

        node_ids = [node.node_id for node in nodes]
        positions = np.asarray([node.position for node in nodes], dtype=np.float32)
        colors = np.asarray([node.color for node in nodes], dtype=np.uint8)
        labels = [
            f"{node.label} #{node.node_id.rsplit('_', 1)[-1]} "
            f"({node.confidence:.2f}, n={node.observations})"
            for node in nodes
        ]
        edges = self.scene_graph.edges()
        edge_pairs = [(edge["source"], edge["target"]) for edge in edges]

        rr.log(
            "scene_graph",
            rr.GraphNodes(
                node_ids=node_ids,
                positions=positions[:, [0, 2]],
                colors=colors,
                labels=labels,
                show_labels=True,
            ),
            rr.GraphEdges(edges=edge_pairs, graph_type="undirected"),
        )
        rr.log(
            "world/scene_graph/nodes",
            rr.Points3D(
                positions,
                colors=colors,
                labels=labels,
                show_labels=True,
                radii=rr.Radius.ui_points(6.0),
            ),
        )

        edge_lookup = {node.node_id: node.position for node in nodes}
        edge_path = "world/scene_graph/relations"
        if edges:
            strips = [
                np.stack([edge_lookup[edge["source"]], edge_lookup[edge["target"]]])
                for edge in edges
            ]
            rr.log(
                edge_path,
                rr.LineStrips3D(
                    strips,
                    colors=[160, 160, 160],
                    labels=["near"] * len(strips),
                    radii=rr.Radius.ui_points(1.5),
                ),
            )
        else:
            rr.log(edge_path, rr.Clear(recursive=False))

    def join(self):
        if self.closed:
            return

        if self.dense_map is not None:
            self.dense_updates = self.dense_map.finalize(self.num_frames)
        if self.frame_index > 0:
            self._log_state(self.frame_index - 1)
        if self.scene_graph is not None and self.scene_graph_output is not None:
            self.scene_graph.save(self.scene_graph_output)
            print(f"Saved scene graph to {self.scene_graph_output}")
        if self.dense_map is not None and self.dense_map_output is not None:
            metadata = self.dense_map.save(self.dense_map_output, self.num_frames)
            print(
                f"Saved dense map to {self.dense_map_output} "
                f"({metadata['keyframes']} keyframes, {metadata['points']} points)"
            )
        if self.sam_segmenter is not None and self.sam_output is not None:
            summary = self.sam_segmenter.save_report(self.sam_output, self.image, self.detections)
            print(f"Saved SAM segmentation report to {self.sam_output}: {summary}")
        rr.disconnect()
        self.closed = True

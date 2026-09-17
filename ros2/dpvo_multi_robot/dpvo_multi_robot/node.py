from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import signal
import threading
import time
import uuid

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, CompressedImage
from std_msgs.msg import Bool, Header, String
import torch

from dpvo import projective_ops as pops
from dpvo.config import cfg as default_cfg
from dpvo.dpvo import DPVO
from dpvo.lietorch import SE3
from dpvo.loop_closure.distributed import DistributedLongTermLoopClosure
from dpvo.loop_closure.tracking_artifact import (
    save_tracking_artifact,
    write_incomplete_manifest,
)
from dpvo.map_gauge import transform_poses_xyzw, transform_points

from .transport import Ros2DistributedTransport


class MultiRobotDpvoNode(Node):
    def __init__(self):
        super().__init__("dpvo_multi_robot")
        self._declare_parameters()

        self.robot_id = self.get_parameter("robot_id").value
        configured_seed = int(self.get_parameter("random_seed").value)
        if configured_seed >= 0:
            np.random.seed(configured_seed % (2**32))
            torch.manual_seed(configured_seed)
            torch.cuda.manual_seed_all(configured_seed)
        self.random_seed = int(torch.initial_seed())
        requested_session = self.get_parameter("session_id").value
        self.session_id = requested_session or (
            f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        )
        self._closed = False
        self.shutdown_thread = None
        self.processing_lock = threading.RLock()
        self.callback_group = ReentrantCallbackGroup()
        self.camera = None
        self._pending_camera_image = None
        self.slam = None
        self.last_published_keyframe = -1
        self.path = PathMessage()
        self.path.header.frame_id = self._frame_id()
        artifact_root = self.get_parameter("tracking_artifact_output").value
        self.tracking_artifact_dir = (
            Path(artifact_root).expanduser() / self.robot_id
            if artifact_root
            else None
        )
        if self.tracking_artifact_dir is not None:
            (self.tracking_artifact_dir / "frames").mkdir(
                parents=True, exist_ok=True
            )
            write_incomplete_manifest(
                self.tracking_artifact_dir,
                robot_id=self.robot_id,
                session_id=self.session_id,
            )

        self.cfg = default_cfg.clone()
        self.cfg.merge_from_file(self.get_parameter("config").value)
        self.cfg.CLASSIC_LOOP_CLOSURE = True
        self.cfg.CLASSIC_PGO_USE_THREADS = True
        self.cfg.MAX_EDGE_AGE = self.get_parameter("max_edge_age").value
        self.cfg.LOOP_CLOSURE = self.get_parameter(
            "enable_dpvo_loop_closure"
        ).value
        self.cfg.ORB_VOCAB_PATH = self.get_parameter("orb_vocab").value
        self.cfg.LOOP_RETR_THRESH = self.get_parameter("bow_threshold").value
        self.cfg.MULTI_ROBOT_RETRIEVAL_BACKEND = self.get_parameter(
            "retrieval_backend"
        ).value
        self.cfg.MULTI_ROBOT_MEGALOC_REPO = self.get_parameter(
            "megaloc_repo"
        ).value
        self.cfg.MULTI_ROBOT_MEGALOC_MODEL_ID = self.get_parameter(
            "megaloc_model_id"
        ).value
        self.cfg.MULTI_ROBOT_MEGALOC_DEVICE = self.get_parameter(
            "megaloc_device"
        ).value
        self.cfg.MULTI_ROBOT_MEGALOC_THRESHOLD = self.get_parameter(
            "megaloc_threshold"
        ).value
        self.cfg.MULTI_ROBOT_LOCAL_FEATURE_BACKEND = self.get_parameter(
            "local_feature_backend"
        ).value
        self.cfg.MULTI_ROBOT_XFEAT_REPO = self.get_parameter(
            "xfeat_repo"
        ).value
        self.cfg.MULTI_ROBOT_XFEAT_TOP_K = self.get_parameter(
            "xfeat_top_k"
        ).value
        self.cfg.MULTI_ROBOT_XFEAT_DETECTION_THRESHOLD = self.get_parameter(
            "xfeat_detection_threshold"
        ).value
        self.cfg.MULTI_ROBOT_LIGHTGLUE_MIN_CONFIDENCE = self.get_parameter(
            "lightglue_min_confidence"
        ).value
        self.cfg.MULTI_ROBOT_BOW_REPETITIONS = self.get_parameter(
            "bow_repetitions"
        ).value
        self.cfg.MULTI_ROBOT_BOW_NMS = self.get_parameter(
            "bow_nms_radius"
        ).value
        self.cfg.MULTI_ROBOT_BOW_BACKFILL = self.get_parameter(
            "bow_backfill"
        ).value
        self.cfg.MULTI_ROBOT_RESERVE_INFLIGHT = self.get_parameter(
            "reserve_inflight"
        ).value
        self.cfg.MULTI_ROBOT_TEASER_NOISE_BOUND = self.get_parameter(
            "teaser_noise_bound"
        ).value
        self.cfg.MULTI_ROBOT_TEASER_REQUIRED = self.get_parameter(
            "teaser_required"
        ).value
        self.cfg.MULTI_ROBOT_MIN_INLIERS = self.get_parameter("min_inliers").value
        self.cfg.MULTI_ROBOT_MIN_INLIER_RATIO = self.get_parameter(
            "min_inlier_ratio"
        ).value
        self.cfg.MULTI_ROBOT_MAX_DEPTH = self.get_parameter("max_depth").value
        self.cfg.MULTI_ROBOT_KEYFRAME_MAX_ATTEMPTS = self.get_parameter(
            "keyframe_max_attempts"
        ).value
        self.cfg.MULTI_ROBOT_KEYFRAME_RETRY_DELAY = self.get_parameter(
            "keyframe_retry_delay"
        ).value

        self.transport = Ros2DistributedTransport(
            self,
            self.robot_id,
            self.processing_lock,
            bow_topic=self.get_parameter("bow_topic").value,
            global_descriptor_topic=self.get_parameter(
                "global_descriptor_topic"
            ).value,
            constraint_topic=self.get_parameter("constraint_topic").value,
            service_prefix=self.get_parameter("service_prefix").value,
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_parameter("pose_topic").value,
            20,
        )
        self.path_publisher = self.create_publisher(
            PathMessage,
            self.get_parameter("path_topic").value,
            5,
        )
        ack_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.frame_ack_publisher = self.create_publisher(
            Header,
            self.get_parameter("frame_ack_topic").value,
            ack_qos,
        )
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.dense_mapper = None
        self.dense_publisher = None
        if self.get_parameter("enable_dense_mapping").value:
            from deploy.jetson.dense_mapping import OnlineDenseMapper
            self.dense_mapper = OnlineDenseMapper(
                self.get_parameter("dense_engine").value,
                fps=float(self.get_parameter("dense_fps").value),
                max_points=int(self.get_parameter("dense_max_points").value),
                voxel_size=float(self.get_parameter("dense_voxel_size").value))
            self.dense_publisher = self.create_publisher(
                PointCloud2, "dpvo/dense_points", QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        from deploy.jetson.patch_preview import PatchPreview
        self.patch_preview = PatchPreview()
        self.last_preview = time.monotonic()
        self.preview_frames = 0
        self.preview_publisher = self.create_publisher(
            CompressedImage, 'dpvo/preview/compressed', QoSProfile(
                depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.tracking_status_publisher = self.create_publisher(String, 'dpvo/tracking_status', 1)
        self.last_online_map = 0.0
        self.online_map_publisher = None
        if float(self.get_parameter("online_map_fps").value) > 0:
            self.online_map_publisher = self.create_publisher(
                PointCloud2, "dpvo/points", QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        image_qos = QoSProfile(
            depth=1,
            reliability=(ReliabilityPolicy.BEST_EFFORT
                         if self.get_parameter("image_best_effort").value
                         else ReliabilityPolicy.RELIABLE),
            durability=DurabilityPolicy.VOLATILE,
        )
        info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.camera_subscription = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info,
            info_qos,
            callback_group=self.callback_group,
        )
        self.image_subscription = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self._image,
            image_qos,
            callback_group=self.image_callback_group,
        )
        self.done_subscription = self.create_subscription(
            Bool,
            self.get_parameter("done_topic").value,
            self._player_done,
            ack_qos,
            callback_group=self.callback_group,
        )
        self.distributed_timer = self.create_timer(
            0.05,
            self._process_distributed,
            callback_group=self.callback_group,
        )
        diagnostics_output = self.get_parameter("loop_diagnostics_output").value
        self.loop_diagnostics_output = (
            Path(diagnostics_output).expanduser() if diagnostics_output else None
        )
        self.loop_diagnostics_timer = None
        if self.loop_diagnostics_output is not None:
            self.loop_diagnostics_timer = self.create_timer(
                max(
                    float(self.get_parameter("loop_diagnostics_period").value),
                    0.1,
                ),
                self._write_loop_diagnostics,
                callback_group=self.callback_group,
            )

        inter_robot_enabled = bool(
            self.get_parameter("enable_inter_robot_loop_closure").value
        )
        pipeline = "tracking-only"
        if inter_robot_enabled:
            pipeline = (
                f"{self.cfg.MULTI_ROBOT_RETRIEVAL_BACKEND}->"
                f"{self.cfg.MULTI_ROBOT_LOCAL_FEATURE_BACKEND}->"
                "lightglue->teaser++"
            )
        self.get_logger().info(
            "multi-robot DPVO ready: "
            f"robot={self.robot_id}, session={self.session_id}, "
            f"seed={self.random_seed}, pipeline={pipeline}"
        )

    def _declare_parameters(self):
        self.declare_parameter("robot_id", "robot0")
        self.declare_parameter("random_seed", -1)
        self.declare_parameter("session_id", "")
        self.declare_parameter("network", "dpvo.pth")
        self.declare_parameter("config", "config/fast.yaml")
        self.declare_parameter("orb_vocab", "ORBvoc.txt")
        self.declare_parameter("vocabulary_id", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("image_scale", 1.0)
        self.declare_parameter("viewer", "none")
        self.declare_parameter("trt_encoders", "")
        self.declare_parameter("image_best_effort", False)
        self.declare_parameter("session_frame_ids", False)
        self.declare_parameter("online_max_frames", 0)
        self.declare_parameter("online_map_fps", 0.0)
        self.declare_parameter("online_max_points", 3000)
        self.declare_parameter("online_preview_fps", 2.0)
        self.declare_parameter("online_check_health", False)
        self.declare_parameter("enable_dense_mapping", False)
        self.declare_parameter("dense_engine", "/output/da3-small-238x378/da3.engine")
        self.declare_parameter("dense_fps", 0.5)
        self.declare_parameter("dense_max_points", 50000)
        self.declare_parameter("dense_voxel_size", 0.02)
        self.declare_parameter("rerun_save", "")
        self.declare_parameter("rerun_connect", "")
        self.declare_parameter("rerun_recording_id", "")
        self.declare_parameter("rerun_entity_prefix", "")
        self.declare_parameter("enable_dpvo_loop_closure", True)
        self.declare_parameter("enable_inter_robot_loop_closure", True)
        self.declare_parameter("tracking_artifact_output", "")
        self.declare_parameter("done_topic", "dpvo/player_done")
        self.declare_parameter("exit_on_player_done", False)
        self.declare_parameter("max_edge_age", 48)
        self.declare_parameter("bow_threshold", 0.04)
        self.declare_parameter("retrieval_backend", "megaloc")
        self.declare_parameter("megaloc_repo", "gmberton/MegaLoc")
        self.declare_parameter("megaloc_model_id", "gmberton/MegaLoc")
        self.declare_parameter("megaloc_device", "cuda")
        self.declare_parameter("megaloc_threshold", 0.20)
        self.declare_parameter("local_feature_backend", "xfeat")
        self.declare_parameter("xfeat_repo", "verlab/accelerated_features")
        self.declare_parameter("xfeat_top_k", 2048)
        self.declare_parameter("xfeat_detection_threshold", 0.05)
        self.declare_parameter("lightglue_min_confidence", 0.10)
        self.declare_parameter("bow_repetitions", 3)
        self.declare_parameter("bow_nms_radius", 50)
        self.declare_parameter("bow_backfill", True)
        self.declare_parameter("reserve_inflight", True)
        self.declare_parameter("teaser_noise_bound", 0.10)
        self.declare_parameter("teaser_required", True)
        self.declare_parameter("min_inliers", 30)
        self.declare_parameter("min_inlier_ratio", 0.20)
        self.declare_parameter("max_depth", 20.0)
        self.declare_parameter("keyframe_max_attempts", 5)
        self.declare_parameter("keyframe_retry_delay", 0.5)
        self.declare_parameter("loop_diagnostics_output", "")
        self.declare_parameter("loop_diagnostics_period", 5.0)
        self.declare_parameter("bow_topic", "/dpvo_multi_robot/bow")
        self.declare_parameter(
            "global_descriptor_topic",
            "/dpvo_multi_robot/global_descriptor",
        )
        self.declare_parameter(
            "constraint_topic", "/dpvo_multi_robot/loop_closure"
        )
        self.declare_parameter("service_prefix", "/dpvo_multi_robot")
        self.declare_parameter("pose_topic", "dpvo/pose")
        self.declare_parameter("path_topic", "dpvo/path")
        self.declare_parameter("frame_ack_topic", "dpvo/frame_ack")
        self.declare_parameter("map_frame", "map")

    def _camera_info(self, message):
        camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(message.d, dtype=np.float64)
        with self.processing_lock:
            self.camera = (camera_matrix, distortion)
            pending = self._pending_camera_image
            self._pending_camera_image = None
            if pending is not None:
                self._image(pending)

    def _process_distributed(self):
        if self.slam is None or not self.processing_lock.acquire(blocking=False):
            return
        try:
            self.slam.long_term_lc.process_transport()
        finally:
            self.processing_lock.release()

    def _player_done(self, message):
        if not message.data or not self.get_parameter("exit_on_player_done").value:
            return
        if self.shutdown_thread is not None:
            return
        self.get_logger().info("Player complete; finalizing tracking artifact")
        self.shutdown_thread = threading.Thread(
            target=self._finalize_and_shutdown,
            daemon=True,
        )
        self.shutdown_thread.start()

    def _finalize_and_shutdown(self):
        try:
            self.close()
        finally:
            if self.context.ok():
                rclpy.shutdown(context=self.context)

    def _write_loop_diagnostics(self, force=False):
        if self.slam is None or self.loop_diagnostics_output is None:
            return
        acquired = self.processing_lock.acquire(blocking=force)
        if not acquired:
            return
        try:
            snapshot = self.slam.long_term_lc.diagnostics_snapshot()
            snapshot.update(
                {
                    "generated_at_unix_ns": time.time_ns(),
                    "keyframes": int(self.slam.n),
                    "parameters": {
                        "retrieval_backend": str(
                            self.cfg.MULTI_ROBOT_RETRIEVAL_BACKEND
                        ),
                        "random_seed": self.random_seed,
                        "megaloc_model_id": str(
                            self.cfg.MULTI_ROBOT_MEGALOC_MODEL_ID
                        ),
                        "megaloc_threshold": float(
                            self.cfg.MULTI_ROBOT_MEGALOC_THRESHOLD
                        ),
                        "local_feature_backend": str(
                            self.cfg.MULTI_ROBOT_LOCAL_FEATURE_BACKEND
                        ),
                        "xfeat_top_k": int(
                            self.cfg.MULTI_ROBOT_XFEAT_TOP_K
                        ),
                        "xfeat_detection_threshold": float(
                            self.cfg.MULTI_ROBOT_XFEAT_DETECTION_THRESHOLD
                        ),
                        "lightglue_min_confidence": float(
                            self.cfg.MULTI_ROBOT_LIGHTGLUE_MIN_CONFIDENCE
                        ),
                        "bow_threshold": float(self.cfg.LOOP_RETR_THRESH),
                        "bow_repetitions": int(
                            self.cfg.MULTI_ROBOT_BOW_REPETITIONS
                        ),
                        "bow_nms_radius": int(self.cfg.MULTI_ROBOT_BOW_NMS),
                        "max_depth": float(self.cfg.MULTI_ROBOT_MAX_DEPTH),
                        "min_inliers": int(self.cfg.MULTI_ROBOT_MIN_INLIERS),
                        "min_inlier_ratio": float(
                            self.cfg.MULTI_ROBOT_MIN_INLIER_RATIO
                        ),
                        "teaser_noise_bound": float(
                            self.cfg.MULTI_ROBOT_TEASER_NOISE_BOUND
                        ),
                        "teaser_required": bool(
                            self.cfg.MULTI_ROBOT_TEASER_REQUIRED
                        ),
                    },
                }
            )
            output = self.loop_diagnostics_output
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(f"{output}.tmp")
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
            temporary.replace(output)
        except Exception as error:
            self.get_logger().warning(
                f"Failed to write loop diagnostics: {error}"
            )
        finally:
            self.processing_lock.release()

    @staticmethod
    def _decode_image(message):
        data = np.frombuffer(message.data, dtype=np.uint8)
        if message.encoding in ("mono8", "8UC1"):
            image = data.reshape(message.height, message.step)[:, : message.width]
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if message.encoding in ("bgr8", "rgb8"):
            image = data.reshape(message.height, message.step)[:, : message.width * 3]
            image = image.reshape(message.height, message.width, 3)
            if message.encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        raise ValueError(f"unsupported image encoding: {message.encoding}")

    @torch.no_grad()
    def _image(self, message):
        with self.processing_lock:
            if self.camera is None:
                # CameraInfo and Image are different topics: their callbacks
                # can arrive in either order. Hold the image and backpressure
                # the player until it is processed, rather than skipping it.
                self._pending_camera_image = message
                return
            camera_matrix, distortion = self.camera
            image = self._decode_image(message)
            if distortion.size and np.any(distortion):
                image = cv2.undistort(image, camera_matrix, distortion)

            scale = self.get_parameter("image_scale").value
            if not 0.0 < scale <= 1.0:
                raise ValueError("image_scale must be in (0, 1]")
            if scale != 1.0:
                image = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            height, width = image.shape[:2]
            image = image[: height - height % 16, : width - width % 16]
            intrinsics_np = np.array(
                [
                    camera_matrix[0, 0],
                    camera_matrix[1, 1],
                    camera_matrix[0, 2],
                    camera_matrix[1, 2],
                ],
                dtype=np.float32,
            ) * scale
            image_tensor = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .cuda()
            )
            intrinsics = torch.from_numpy(intrinsics_np).cuda()

            if self.slam is None:
                self.slam = self._create_slam(image_tensor.shape[1:])

            timestamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            if self.tracking_artifact_dir is not None:
                input_index = int(self.slam.counter)
                frame_path = (
                    self.tracking_artifact_dir
                    / "frames"
                    / f"{input_index:08d}.jpg"
                )
                if not cv2.imwrite(
                    str(frame_path),
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                ):
                    raise RuntimeError(f"failed to write tracking frame {frame_path}")
            frame_limit = int(self.get_parameter("online_max_frames").value)
            if frame_limit and (self.slam.counter >= frame_limit or self.slam.n >= self.slam.N - 2):
                raise RuntimeError("Live session capacity reached; press Start for a fresh session")
            if self.dense_mapper is not None:
                self.dense_mapper.remember(int(self.slam.counter), image)
            preview_input_index = int(self.slam.counter)
            self.slam(timestamp, image_tensor, intrinsics)
            if self.get_parameter("online_check_health").value:
                from deploy.jetson.tracking_health import tracking_health
                health = tracking_health(self.slam.pg.poses_[:self.slam.n],
                    self.slam.pg.patches_[max(0, self.slam.n-3):self.slam.n, :, 2],
                    self.slam.is_initialized)
                if health['lost']:
                    raise RuntimeError("Tracking lost; press Start for a fresh session")
            if self.dense_mapper is not None:
                try:
                    cloud = self.dense_mapper.update(self.slam, intrinsics_np)
                    if cloud is not None:
                        self._publish_point_cloud(message, *cloud, self.dense_publisher)
                        self.get_logger().info(
                            f"DA3 dense: {len(cloud[0])} points; {self.dense_mapper.stats}",
                            throttle_duration_sec=5.)
                except Exception as error:
                    self.get_logger().error(f"Dense mapping stopped: {error}")
                    self.dense_mapper.close()
                    self.dense_mapper = None
            try:
                self._publish_preview(message, image, preview_input_index)
            except Exception as error:
                self.get_logger().warning(f"Patch preview failed: {error}", throttle_duration_sec=10.)
            self._publish_online_map(message)
            self._publish_pose(message)
            self._publish_ack(message)

    def _publish_preview(self, image_message, image, input_index):
        self.preview_frames += 1
        now = time.monotonic()
        rate = float(self.get_parameter('online_preview_fps').value)
        if rate <= 0:
            return
        due = now - self.last_preview >= 1 / rate
        # Observe every processed frame; JPEG publication remains rate-limited.
        preview, tracks = self.patch_preview.snapshot(self.slam, image, input_index, render=due)
        if not due:
            return
        elapsed = now - self.last_preview
        ok, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            message = CompressedImage()
            message.header.stamp = image_message.header.stamp
            message.header.frame_id = self._frame_id()
            message.format = 'jpeg'
            message.data = jpeg.tobytes()
            self.preview_publisher.publish(message)
        self.tracking_status_publisher.publish(String(data=json.dumps(dict(
            session=self.session_id, state='tracking' if self.slam.is_initialized else 'initializing',
            processed_frames=int(self.slam.counter), keyframes=int(self.slam.n),
            points=int(self.slam.m) if self.slam.is_initialized else 0,
            patch_tracks=tracks, processing_fps=self.preview_frames / elapsed))))
        self.last_preview, self.preview_frames = now, 0

    def _publish_ack(self, image_message):
        acknowledgement = Header()
        acknowledgement.stamp = image_message.header.stamp
        acknowledgement.frame_id = self.robot_id
        self.frame_ack_publisher.publish(acknowledgement)

    def _create_slam(self, image_shape):
        vocabulary_id = self.get_parameter("vocabulary_id").value or None

        def loop_closure_factory(cfg, patchgraph):
            return DistributedLongTermLoopClosure(
                cfg,
                patchgraph,
                transport=self.transport,
                robot_id=self.robot_id,
                session_id=self.session_id,
                vocabulary_id=vocabulary_id,
                distributed_enabled=bool(
                    self.get_parameter("enable_inter_robot_loop_closure").value
                ),
            )

        viewer = self.get_parameter("viewer").value
        if viewer == "none":
            viewer = None
        rerun_save = self.get_parameter("rerun_save").value or None
        rerun_connect = self.get_parameter("rerun_connect").value or None
        if rerun_save and rerun_connect:
            raise ValueError("rerun_save and rerun_connect are mutually exclusive")
        if (rerun_save or rerun_connect) and viewer is None:
            viewer = "rerun"

        slam = DPVO(
            self.cfg,
            self.get_parameter("network").value,
            ht=image_shape[0],
            wd=image_shape[1],
            viewer=viewer,
            viewer_output=Path(rerun_save) if rerun_save else None,
            viewer_connect=rerun_connect,
            viewer_recording_id=(
                self.get_parameter("rerun_recording_id").value or None
            ),
            viewer_entity_prefix=(
                self.get_parameter("rerun_entity_prefix").value or None
            ),
            long_term_lc_factory=loop_closure_factory,
        )
        engines = self.get_parameter("trt_encoders").value
        if engines:
            from deploy.jetson.tensorrt_encoder import install_encoders
            install_encoders(slam.network, engines, self.get_parameter("network").value)
        return slam

    def _frame_id(self):
        if self.get_parameter("session_frame_ids").value:
            from .online_common import session_frame
            return session_frame(self.robot_id, self.session_id)
        return self.get_parameter("map_frame").value

    def _publish_online_map(self, image_message):
        if self.online_map_publisher is None or not self.slam.is_initialized:
            return
        now = time.monotonic()
        if now - self.last_online_map < 1.0 / float(self.get_parameter("online_map_fps").value):
            return
        self.last_online_map = now
        count = self.slam.m
        limit = max(1, int(self.get_parameter("online_max_points").value))
        step = max(1, (count + limit - 1) // limit)
        points = self.slam.pg.points_[:count:step].detach().float().cpu().numpy()
        points = transform_points(self.slam.pg.session_from_map_, points)
        colors = self.slam.pg.colors_.reshape(-1, 3)[:count:step].detach().cpu().numpy()
        self._publish_point_cloud(image_message, points, colors, self.online_map_publisher)

    def _publish_point_cloud(self, image_message, points, colors, publisher):
        valid = np.isfinite(points).all(axis=1)
        points, colors = points[valid], colors[valid].astype(np.uint32)
        packed = np.zeros(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
        for axis, name in enumerate(('x', 'y', 'z')):
            packed[name] = points[:, axis]
        packed['rgb'] = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        message = PointCloud2()
        message.header.stamp = image_message.header.stamp
        message.header.frame_id = self._frame_id()
        message.height, message.width = 1, len(points)
        message.fields = [PointField(name=name, offset=4*i, count=1,
                          datatype=PointField.FLOAT32 if i < 3 else PointField.UINT32)
                          for i, name in enumerate(('x', 'y', 'z', 'rgb'))]
        message.point_step, message.row_step = 16, 16 * len(points)
        message.is_dense = True
        message.data = packed.tobytes()
        publisher.publish(message)

    def _publish_pose(self, image_message):
        if self.slam.n <= 0 or self.slam.n == self.last_published_keyframe:
            return
        self.last_published_keyframe = self.slam.n
        poses = (
            SE3(self.slam.pg.poses_[: self.slam.n])
            .inv()
            .data.detach()
            .float()
            .cpu()
            .numpy()
        )
        poses = transform_poses_xyzw(
            self.slam.pg.session_from_map_,
            poses,
        )
        pose_data = poses[-1]
        pose = PoseStamped()
        pose.header.stamp = image_message.header.stamp
        pose.header.frame_id = self._frame_id()
        pose.pose.position.x = float(pose_data[0])
        pose.pose.position.y = float(pose_data[1])
        pose.pose.position.z = float(pose_data[2])
        pose.pose.orientation.x = float(pose_data[3])
        pose.pose.orientation.y = float(pose_data[4])
        pose.pose.orientation.z = float(pose_data[5])
        pose.pose.orientation.w = float(pose_data[6])
        self.pose_publisher.publish(pose)

        # Rebuild the path because DPVO bundle adjustment updates historical
        # poses. All poses are expressed in the stable session map gauge.
        self.path = PathMessage()
        self.path.header.stamp = pose.header.stamp
        self.path.header.frame_id = self._frame_id()
        for keyframe_id, pose_data in enumerate(poses):
            path_pose = PoseStamped()
            path_pose.header.frame_id = self.path.header.frame_id
            input_index = int(self.slam.pg.tstamps_[keyframe_id])
            timestamp_ns = int(round(self.slam.tlist[input_index] * 1_000_000_000))
            path_pose.header.stamp.sec = timestamp_ns // 1_000_000_000
            path_pose.header.stamp.nanosec = timestamp_ns % 1_000_000_000
            path_pose.pose.position.x = float(pose_data[0])
            path_pose.pose.position.y = float(pose_data[1])
            path_pose.pose.position.z = float(pose_data[2])
            path_pose.pose.orientation.x = float(pose_data[3])
            path_pose.pose.orientation.y = float(pose_data[4])
            path_pose.pose.orientation.z = float(pose_data[5])
            path_pose.pose.orientation.w = float(pose_data[6])
            self.path.poses.append(path_pose)
        self.path_publisher.publish(self.path)

    def _export_tracking_artifact(self, slam, input_poses):
        if self.tracking_artifact_dir is None:
            return
        count = int(slam.n)
        if count < 1:
            raise RuntimeError("cannot export an empty DPVO tracking artifact")
        internal_poses = (
            slam.pg.poses_[:count].detach().float().cpu().numpy()
        )
        local_poses = (
            SE3(slam.pg.poses_[:count])
            .inv()
            .data.detach()
            .float()
            .cpu()
            .numpy()
        )
        local_poses = transform_poses_xyzw(
            slam.pg.session_from_map_, local_poses
        ).astype(np.float32)
        input_indices = slam.pg.tstamps_[:count].astype(np.int64, copy=True)
        timestamps = np.asarray(
            [slam.tlist[int(index)] for index in input_indices],
            dtype=np.float64,
        )
        patch_disparities = (
            slam.pg.patches_[:count, :, 2, 1, 1]
            .median(dim=1)
            .values.detach()
            .float()
            .cpu()
            .numpy()
        )
        with torch.no_grad():
            points = pops.point_cloud(
                SE3(slam.poses),
                slam.patches[:, : slam.m],
                slam.intrinsics,
                slam.ix[: slam.m],
            )
            points = (
                points[..., 1, 1, :3] / points[..., 1, 1, 3:]
            ).reshape(-1, 3)
            points = points.detach().float().cpu().numpy()
        colors = (
            slam.pg.colors_[:count]
            .reshape(-1, 3)
            .detach()
            .cpu()
            .numpy()
        )
        if len(points) != count * slam.M or len(colors) != len(points):
            raise RuntimeError(
                f"invalid final sparse map: {len(points)} points, "
                f"{len(colors)} colors, {count} keyframes, M={slam.M}"
            )
        manifest = save_tracking_artifact(
            self.tracking_artifact_dir,
            robot_id=self.robot_id,
            session_id=self.session_id,
            keyframe_input_indices=input_indices,
            keyframe_timestamps=timestamps,
            keyframe_poses_xyzw=local_poses,
            internal_poses_xyzw=internal_poses,
            internal_intrinsics=(
                slam.pg.intrinsics_[:count].detach().float().cpu().numpy()
            ),
            patch_disparities=patch_disparities,
            session_from_map=slam.pg.session_from_map_,
            input_poses_xyzw=input_poses,
            map_points=points,
            map_colors=colors,
            patches_per_keyframe=slam.M,
            image_width=slam.wd,
            image_height=slam.ht,
            dpvo_resolution=slam.RES,
            input_frame_count=slam.counter,
            random_seed=self.random_seed,
            config_path=self.get_parameter("config").value,
            network_path=self.get_parameter("network").value,
        )
        self.get_logger().info(
            f"Saved {count}-keyframe tracking artifact: {manifest}"
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.dense_mapper is not None:
            self.dense_mapper.close()
            self.dense_mapper = None
        if self.slam is not None:
            with self.processing_lock:
                # Online BA has already run. Stage one saves the final local
                # graph but deliberately performs no inter-robot verification.
                slam = self.slam
                input_poses, _ = slam.terminate(final_updates=0)
                self._write_loop_diagnostics(force=True)
                self._export_tracking_artifact(slam, input_poses)
                self.slam = None


def main(args=None):
    faulthandler_enabled = os.environ.get("DPVO_FAULTHANDLER", "").lower()
    if faulthandler_enabled in ("1", "true", "yes"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    rclpy.init(args=args)
    node = MultiRobotDpvoNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

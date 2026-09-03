"""Pose-estimated Depth Anything 3 reconstruction for staged CBS runs.

The staged tracker stores camera-to-world poses in a Sim(3) pose graph.  Each
CBS pose is projected to rigid SE(3), preserving its camera centre and keeping
its Sim(3) scale only as diagnostic metadata.  DA3 receives the known
intrinsics but no CBS extrinsics, estimates its own two-view poses, and the
ratio of CBS to DA3 camera baselines converts DA3 depth into the CBS rigid-pose
gauge.  Scaled depths are finally backprojected with the rigid CBS poses.

The public functions in this file are intentionally independent of DA3 where
possible.  Pair construction, cache validation, filtering, and fusion can all
be tested without a GPU or model download.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import types
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .centralized import Sim3
from .pose_graph import Sim3PoseGraph, read_json
from .tracking_artifact import TrackingArtifact, load_artifact_root


FORMAT_NAME = "dpvo_cbs_da3_dense_map"
FORMAT_VERSION = 4
CACHE_FORMAT_NAME = "dpvo_cbs_da3_pose_estimated_baseline_scaled_pair_cache"
CACHE_FORMAT_VERSION = 2
INFERENCE_MODE = "da3_pose_estimated_cbs_baseline_scaled_v1"
DEFAULT_MODEL = "depth-anything/DA3-LARGE"
DEFAULT_CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DEFAULT_WEIGHTS_REVISION = "c54c26b16ec04d218e8d584ecf4bce082a9fcc20"


def robot_sort_key(robot_id: str) -> tuple[int, int | str]:
    suffix = robot_id.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot_id)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True, order=True)
class FrameKey:
    robot_id: str
    session_id: str
    keyframe_id: int

    def label(self) -> str:
        return f"{self.robot_id}:{self.session_id}:{self.keyframe_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "session_id": self.session_id,
            "keyframe_id": self.keyframe_id,
        }


@dataclass(frozen=True)
class CbsPose:
    vertex_id: int
    key: FrameKey
    camera_from_world: np.ndarray
    camera_center: np.ndarray
    rotation_camera_to_world: np.ndarray
    scale: float


@dataclass(frozen=True)
class PairJob:
    job_id: str
    kind: str
    views: tuple[FrameKey, FrameKey]
    contributing_views: tuple[bool, bool]
    source_edge_id: int | None = None
    fallback_from: tuple[FrameKey, FrameKey] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "views": [view.to_dict() for view in self.views],
            "contributing_views": list(self.contributing_views),
            "source_edge_id": self.source_edge_id,
        }
        if self.fallback_from is not None:
            result["fallback_from"] = [view.to_dict() for view in self.fallback_from]
        return result


@dataclass(frozen=True)
class InferenceSettings:
    model: str = DEFAULT_MODEL
    code_revision: str = DEFAULT_CODE_REVISION
    weights_revision: str = DEFAULT_WEIGHTS_REVISION
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    minimum_da3_baseline: float = 1e-8
    inference_mode: str = INFERENCE_MODE
    pose_estimation: str = "camera_decoder"
    input_extrinsics: bool = False
    known_intrinsics: bool = True
    align_to_input_ext_scale: bool = False
    infer_gs: bool = False
    use_ray_pose: bool = False

    def validate(self) -> "InferenceSettings":
        if self.inference_mode != INFERENCE_MODE:
            raise ValueError(f"unsupported dense inference mode: {self.inference_mode!r}")
        if self.pose_estimation != "camera_decoder":
            raise ValueError("corrected dense inference requires DA3 camera-decoder pose estimation")
        if self.input_extrinsics or self.align_to_input_ext_scale:
            raise ValueError("corrected dense inference forbids CBS extrinsic pose conditioning/alignment")
        if not self.known_intrinsics:
            raise ValueError("corrected dense inference requires known camera intrinsics")
        if self.infer_gs or self.use_ray_pose:
            raise ValueError("corrected dense inference requires the non-Gaussian camera-decoder branch")
        if self.process_res < 14:
            raise ValueError("DA3 process resolution must be at least 14")
        if not np.isfinite(self.minimum_da3_baseline) or self.minimum_da3_baseline <= 0.0:
            raise ValueError("minimum DA3 baseline must be finite and positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "code_revision": self.code_revision,
            "weights_revision": self.weights_revision,
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "minimum_da3_baseline": self.minimum_da3_baseline,
            "inference_mode": self.inference_mode,
            "pose_estimation": self.pose_estimation,
            "input_extrinsics": self.input_extrinsics,
            "known_intrinsics": self.known_intrinsics,
            "align_to_input_ext_scale": self.align_to_input_ext_scale,
            "infer_gs": self.infer_gs,
            "use_ray_pose": self.use_ray_pose,
        }


@dataclass(frozen=True)
class FusionSettings:
    confidence_percentile: float = 40.0
    far_depth_percentile: float = 99.5
    reprojection_relative_tolerance: float = 0.10
    require_reprojection_overlap: bool = False
    depth_scale_refinement: str = "none"
    dpvo_keypoint_confidence_percentile: float = 50.0
    dpvo_keypoint_min_depth: float = 0.05
    dpvo_keypoint_max_depth: float = 20.0
    dpvo_keypoint_min_matches_per_view: int = 8
    dpvo_keypoint_mad_multiplier: float = 3.0
    pixel_stride: int = 4
    voxel_size: float = 0.02

    def validate(self) -> "FusionSettings":
        if not 0.0 <= self.confidence_percentile <= 100.0:
            raise ValueError("confidence percentile must be in [0, 100]")
        if not 0.0 < self.far_depth_percentile <= 100.0:
            raise ValueError("far-depth percentile must be in (0, 100]")
        if self.reprojection_relative_tolerance < 0.0:
            raise ValueError("reprojection tolerance must be non-negative")
        if not isinstance(self.require_reprojection_overlap, (bool, np.bool_)):
            raise ValueError("require_reprojection_overlap must be boolean")
        if self.depth_scale_refinement not in ("none", "dpvo_keypoints"):
            raise ValueError(
                "depth_scale_refinement must be 'none' or 'dpvo_keypoints'"
            )
        if not 0.0 <= self.dpvo_keypoint_confidence_percentile <= 100.0:
            raise ValueError("DPVO-keypoint confidence percentile must be in [0, 100]")
        if (
            not np.isfinite(self.dpvo_keypoint_min_depth)
            or not np.isfinite(self.dpvo_keypoint_max_depth)
            or self.dpvo_keypoint_min_depth <= 0.0
            or self.dpvo_keypoint_max_depth <= self.dpvo_keypoint_min_depth
        ):
            raise ValueError("invalid DPVO-keypoint depth interval")
        if self.dpvo_keypoint_min_matches_per_view < 1:
            raise ValueError("DPVO-keypoint minimum matches must be positive")
        if (
            not np.isfinite(self.dpvo_keypoint_mad_multiplier)
            or self.dpvo_keypoint_mad_multiplier <= 0.0
        ):
            raise ValueError("DPVO-keypoint MAD multiplier must be positive")
        if self.pixel_stride < 1:
            raise ValueError("pixel stride must be at least one")
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("voxel size must be finite and positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_percentile": self.confidence_percentile,
            "far_depth_percentile": self.far_depth_percentile,
            "reprojection_relative_tolerance": self.reprojection_relative_tolerance,
            "require_reprojection_overlap": bool(self.require_reprojection_overlap),
            "depth_scale_refinement": self.depth_scale_refinement,
            "dpvo_keypoint_confidence_percentile": self.dpvo_keypoint_confidence_percentile,
            "dpvo_keypoint_min_depth": self.dpvo_keypoint_min_depth,
            "dpvo_keypoint_max_depth": self.dpvo_keypoint_max_depth,
            "dpvo_keypoint_min_matches_per_view": self.dpvo_keypoint_min_matches_per_view,
            "dpvo_keypoint_mad_multiplier": self.dpvo_keypoint_mad_multiplier,
            "pixel_stride": self.pixel_stride,
            "voxel_size_cbs_units": self.voxel_size,
        }


@dataclass
class PairPrediction:
    depth: np.ndarray
    confidence: np.ndarray
    processed_rgb: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray

    def normalized(self) -> "PairPrediction":
        depth = np.asarray(self.depth, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        rgb = np.asarray(self.processed_rgb)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        extrinsics = np.asarray(self.extrinsics, dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if confidence.ndim == 4 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        if depth.ndim == 2:
            depth = depth[None]
        if confidence.ndim == 2:
            confidence = confidence[None]
        if rgb.ndim == 3:
            rgb = rgb[None]
        if depth.shape[0] != 2 or confidence.shape != depth.shape:
            raise ValueError(
                f"DA3 returned depth/confidence shapes {depth.shape}/{confidence.shape}; "
                "expected (2, H, W)"
            )
        if rgb.shape != (*depth.shape, 3):
            raise ValueError(
                f"DA3 returned RGB shape {rgb.shape}; expected {(*depth.shape, 3)}"
            )
        if intrinsics.shape != (2, 3, 3):
            raise ValueError(f"DA3 returned intrinsics shape {intrinsics.shape}")
        if extrinsics.shape == (2, 3, 4):
            padded = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
            padded[:, :3, :] = extrinsics
            extrinsics = padded
        if extrinsics.shape != (2, 4, 4):
            raise ValueError(f"DA3 returned extrinsics shape {extrinsics.shape}")
        if not np.issubdtype(rgb.dtype, np.integer):
            maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
            if maximum <= 1.0:
                rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if not np.isfinite(intrinsics).all():
            raise ValueError("DA3 returned non-finite adjusted intrinsics")
        return PairPrediction(depth, confidence, rgb, intrinsics, extrinsics)


class InferenceBackend(Protocol):
    @property
    def revisions(self) -> dict[str, Any]: ...

    def infer(
        self,
        image_paths: Sequence[Path],
        intrinsics: np.ndarray,
        settings: InferenceSettings,
    ) -> PairPrediction: ...


def recover_full_resolution_intrinsics(
    internal_intrinsics: np.ndarray, dpvo_resolution: int
) -> np.ndarray:
    """Recover a 3x3 image-space K from DPVO's downsampled fx,fy,cx,cy."""

    values = np.asarray(internal_intrinsics, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"intrinsics must end in four values, got {values.shape}")
    if int(dpvo_resolution) < 1:
        raise ValueError("DPVO resolution must be positive")
    scaled = values * float(dpvo_resolution)
    output = np.zeros((*scaled.shape[:-1], 3, 3), dtype=np.float64)
    output[..., 0, 0] = scaled[..., 0]
    output[..., 1, 1] = scaled[..., 1]
    output[..., 0, 2] = scaled[..., 2]
    output[..., 1, 2] = scaled[..., 3]
    output[..., 2, 2] = 1.0
    if not np.isfinite(output).all() or np.any(output[..., (0, 1), (0, 1)] <= 0):
        raise ValueError("recovered intrinsics are non-finite or non-positive")
    return output


def project_rotation_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest proper rotation, discarding any embedded scale/shear."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("CBS rotation must be a finite 3x3 matrix")
    left, singular_values, right_t = np.linalg.svd(matrix)
    if not np.isfinite(singular_values).all() or float(singular_values[-1]) <= 1e-12:
        raise ValueError("CBS rotation block is singular and cannot be projected to SO(3)")
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("CBS rotation projection did not produce a proper rotation")
    return rotation


def cbs_sim3_to_rigid_w2c(pose: Sim3) -> np.ndarray:
    """Project ``T_world_camera`` Sim(3) to SE(3) without moving its centre.

    ``pose.translation`` is the camera centre in the CBS world frame for the
    repository's ``p_world = s R p_camera + t`` convention.  The Sim(3) scale
    is deliberately absent from both the rigid rotation and translation.
    """

    centre = np.asarray(pose.translation, dtype=np.float64)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ValueError("CBS camera centre must be a finite 3-vector")
    if not np.isfinite(pose.scale) or pose.scale <= 0.0:
        raise ValueError("CBS Sim(3) scale must be finite and positive")
    camera_to_world_rotation = project_rotation_to_so3(pose.rotation)
    world_to_camera_rotation = camera_to_world_rotation.T
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = world_to_camera_rotation
    world_to_camera[:3, 3] = -(world_to_camera_rotation @ centre)
    return world_to_camera


def camera_centers_from_world_to_camera(extrinsics: np.ndarray) -> np.ndarray:
    """Convert DA3 world-to-camera matrices to camera centres.

    The pinned DA3 camera decoder returns ``(N, 3, 4)`` world-to-camera
    matrices even though its public ``Prediction`` annotation says 4x4.  Both
    source shapes are accepted here.
    """

    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[0] != 2:
        raise ValueError(f"expected two DA3 extrinsics, got {extrinsics.shape}")
    if extrinsics.shape[-2:] == (3, 4):
        homogeneous = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
        homogeneous[:, :3, :] = extrinsics
        extrinsics = homogeneous
    if extrinsics.shape != (2, 4, 4):
        raise ValueError(f"expected DA3 extrinsics shape (2,3,4) or (2,4,4), got {extrinsics.shape}")
    if not np.isfinite(extrinsics).all():
        raise ValueError("DA3 extrinsics contain non-finite values")
    rotations = extrinsics[:, :3, :3]
    errors = np.linalg.norm(
        np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3), axis=(1, 2)
    )
    determinants = np.linalg.det(rotations)
    if np.any(errors > 1e-3) or np.any(np.abs(determinants - 1.0) > 1e-3):
        raise ValueError(
            "DA3 extrinsics are not rigid world-to-camera transforms "
            f"(orthogonality={errors.tolist()}, determinants={determinants.tolist()})"
        )
    return -np.einsum(
        "nij,nj->ni", np.swapaxes(rotations, 1, 2), extrinsics[:, :3, 3]
    )


class InvalidDa3BaselineError(RuntimeError):
    """Raised before caching when DA3 cannot provide a usable pair gauge."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def pair_scale_diagnostics(
    predicted_world_to_camera: np.ndarray,
    cbs_pair: Sequence[CbsPose],
    *,
    minimum_da3_baseline: float = 1e-8,
) -> dict[str, Any]:
    """Compute the DA3-depth-to-CBS-gauge scale and relative-pose diagnostics."""

    if len(cbs_pair) != 2:
        raise ValueError("pair diagnostics require exactly two CBS poses")
    cbs_centres = np.stack([pose.camera_center for pose in cbs_pair]).astype(np.float64)
    cbs_baseline = float(np.linalg.norm(cbs_centres[1] - cbs_centres[0]))
    diagnostics: dict[str, Any] = {
        "cbs_baseline": cbs_baseline,
        "da3_baseline": None,
        "applied_depth_scale": None,
        "relative_rotation_error_degrees": None,
        "translation_direction_error_degrees": None,
        "cbs_source_sim3_scales": [float(pose.scale) for pose in cbs_pair],
        "invalid_da3_baseline_reason": None,
        "da3_extrinsics_convention": "world_to_camera",
    }
    if not np.isfinite(cbs_baseline) or cbs_baseline <= 0.0:
        raise ValueError(f"CBS pair baseline must be finite and positive, got {cbs_baseline}")
    try:
        da3_centres = camera_centers_from_world_to_camera(predicted_world_to_camera)
    except ValueError as error:
        diagnostics["invalid_da3_baseline_reason"] = f"invalid_da3_extrinsics: {error}"
        raise InvalidDa3BaselineError(
            f"cannot recover DA3 camera centres: {error}", diagnostics
        ) from error
    da3_baseline = float(np.linalg.norm(da3_centres[1] - da3_centres[0]))
    diagnostics["da3_baseline"] = da3_baseline if np.isfinite(da3_baseline) else None
    if not np.isfinite(da3_baseline):
        diagnostics["invalid_da3_baseline_reason"] = "non_finite_da3_baseline"
        raise InvalidDa3BaselineError("DA3 predicted a non-finite two-view baseline", diagnostics)
    if da3_baseline <= minimum_da3_baseline:
        diagnostics["invalid_da3_baseline_reason"] = (
            f"da3_baseline_at_or_below_{minimum_da3_baseline:.17g}"
        )
        raise InvalidDa3BaselineError(
            f"DA3 predicted an effectively zero two-view baseline ({da3_baseline:.9g} <= "
            f"{minimum_da3_baseline:.9g})",
            diagnostics,
        )

    scale = cbs_baseline / da3_baseline
    if not np.isfinite(scale) or scale <= 0.0:
        diagnostics["invalid_da3_baseline_reason"] = "non_finite_or_non_positive_depth_scale"
        raise InvalidDa3BaselineError(f"invalid DA3-to-CBS depth scale {scale}", diagnostics)

    da3_ext = np.asarray(predicted_world_to_camera, dtype=np.float64)
    if da3_ext.shape[-2:] == (3, 4):
        da3_h = np.tile(np.eye(4), (2, 1, 1))
        da3_h[:, :3] = da3_ext
        da3_ext = da3_h
    cbs_ext = np.stack([pose.camera_from_world for pose in cbs_pair])
    da3_relative_rotation = da3_ext[1, :3, :3] @ da3_ext[0, :3, :3].T
    cbs_relative_rotation = cbs_ext[1, :3, :3] @ cbs_ext[0, :3, :3].T
    rotation_delta = da3_relative_rotation @ cbs_relative_rotation.T
    rotation_cosine = np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    rotation_error = float(np.degrees(np.arccos(rotation_cosine)))

    da3_direction_camera0 = da3_ext[0, :3, :3] @ (da3_centres[1] - da3_centres[0])
    cbs_direction_camera0 = cbs_ext[0, :3, :3] @ (cbs_centres[1] - cbs_centres[0])
    direction_cosine = np.clip(
        np.dot(da3_direction_camera0, cbs_direction_camera0)
        / (da3_baseline * cbs_baseline),
        -1.0,
        1.0,
    )
    direction_error = float(np.degrees(np.arccos(direction_cosine)))
    diagnostics.update(
        {
            "applied_depth_scale": float(scale),
            "relative_rotation_error_degrees": rotation_error,
            "translation_direction_error_degrees": direction_error,
        }
    )
    return diagnostics


def scale_prediction_to_cbs(
    prediction: PairPrediction,
    cbs_pair: Sequence[CbsPose],
    *,
    minimum_da3_baseline: float = 1e-8,
) -> tuple[PairPrediction, dict[str, Any]]:
    """Scale both DA3 depth maps by the baseline ratio, retaining DA3 poses."""

    prediction = prediction.normalized()
    diagnostics = pair_scale_diagnostics(
        prediction.extrinsics,
        cbs_pair,
        minimum_da3_baseline=minimum_da3_baseline,
    )
    scaled = PairPrediction(
        prediction.depth * diagnostics["applied_depth_scale"],
        prediction.confidence,
        prediction.processed_rgb,
        prediction.intrinsics,
        prediction.extrinsics,
    ).normalized()
    return scaled, diagnostics


def load_cbs_csv(path: Path) -> tuple[dict[FrameKey, CbsPose], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CBS trajectory CSV does not exist: {path}")
    poses: dict[FrameKey, CbsPose] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "vertex_id", "robot_id", "session_id", "keyframe_id",
            "tx", "ty", "tz", "qx", "qy", "qz", "qw", "scale",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"CBS CSV is missing columns {missing}: {path}")
        from scipy.spatial.transform import Rotation

        for row_number, row in enumerate(reader, start=2):
            try:
                key = FrameKey(
                    str(row["robot_id"]), str(row["session_id"]), int(row["keyframe_id"])
                )
                scale = float(row["scale"])
                sim3 = Sim3(
                    [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                    Rotation.from_quat(
                        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
                    ).as_matrix(),
                    scale,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"malformed CBS CSV row {row_number}: {error}") from error
            if key in poses:
                raise ValueError(f"duplicate CBS pose for {key.label()}")
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"invalid CBS scale for {key.label()}: {scale}")
            rigid_world_to_camera = cbs_sim3_to_rigid_w2c(sim3)
            poses[key] = CbsPose(
                vertex_id=int(row["vertex_id"]),
                key=key,
                camera_from_world=rigid_world_to_camera,
                camera_center=sim3.translation.copy(),
                rotation_camera_to_world=rigid_world_to_camera[:3, :3].T.copy(),
                scale=scale,
            )
    if not poses:
        raise ValueError(f"CBS CSV contains no poses: {path}")
    return poses, sha256_file(path)


def _frame_key_from_vertex(vertex: Any) -> FrameKey:
    if vertex.keyframe_id is None:
        raise ValueError(f"graph vertex {vertex.vertex_id} has no keyframe_id")
    return FrameKey(vertex.robot_id, vertex.session_id, int(vertex.keyframe_id))


def validate_inputs(
    artifacts: dict[str, TrackingArtifact],
    graph: Sim3PoseGraph,
    cbs_poses: dict[FrameKey, CbsPose],
    *,
    graph_path: Path | None = None,
    dpgo_provenance_path: Path | None = None,
) -> None:
    """Cross-check every artifact, graph vertex, and CBS estimate."""

    if graph.metadata.get("pipeline_stage") != "geometric_verification":
        raise ValueError(
            "input graph is not the verified geometric-verification graph "
            f"(pipeline_stage={graph.metadata.get('pipeline_stage')!r})"
        )
    if graph.metadata.get("input_contains_global_optimization", True):
        raise ValueError("verified graph is contaminated by global optimization")
    if any(vertex.optimized_estimate is not None for vertex in graph.vertices):
        raise ValueError("verified graph contains optimized vertex estimates")
    loop_count = sum(
        edge.edge_type == "inter_robot_loop_closure" for edge in graph.edges
    )
    declared_loop_count = graph.metadata.get("inter_robot_loop_count")
    if declared_loop_count is not None and int(declared_loop_count) != loop_count:
        raise ValueError(
            "verified graph loop-count metadata mismatch: "
            f"declared {declared_loop_count}, found {loop_count}"
        )
    graph_keys = {_frame_key_from_vertex(vertex) for vertex in graph.vertices}
    if len(graph_keys) != len(graph.vertices):
        raise ValueError("verified graph contains duplicate frame identities")
    artifact_keys: set[FrameKey] = set()
    for robot_id, artifact in artifacts.items():
        if robot_id != artifact.robot_id:
            raise ValueError(f"artifact mapping key {robot_id} does not match manifest")
        artifact_keys.update(
            FrameKey(robot_id, artifact.session_id, keyframe_id)
            for keyframe_id in range(artifact.keyframe_count)
        )
        state_hashes = graph.metadata.get("artifact_state_sha256", {})
        image_hashes = graph.metadata.get("artifact_keyframe_images_sha256", {})
        if state_hashes.get(robot_id) != artifact.manifest.get("state_sha256"):
            raise ValueError(f"verified graph state checksum does not match {robot_id}")
        if image_hashes.get(robot_id) != artifact.manifest.get("keyframe_images_sha256"):
            raise ValueError(f"verified graph image checksum does not match {robot_id}")
    if graph_keys != artifact_keys:
        missing = sorted(artifact_keys - graph_keys)[:3]
        extra = sorted(graph_keys - artifact_keys)[:3]
        raise ValueError(
            "verified graph/tracking keyframe mismatch: "
            f"missing={len(artifact_keys - graph_keys)} {missing}, "
            f"extra={len(graph_keys - artifact_keys)} {extra}"
        )
    missing_cbs = graph_keys - cbs_poses.keys()
    if missing_cbs:
        first = sorted(missing_cbs)[0]
        raise ValueError(
            f"CBS output is missing {len(missing_cbs)} exact graph keyframes; "
            f"first: {first.label()}"
        )
    vertex_by_id = {vertex.vertex_id: vertex for vertex in graph.vertices}
    for key in graph_keys:
        vertex = next(vertex for vertex in graph.vertices if _frame_key_from_vertex(vertex) == key)
        if cbs_poses[key].vertex_id != vertex.vertex_id:
            raise ValueError(
                f"CBS vertex ID mismatch for {key.label()}: "
                f"{cbs_poses[key].vertex_id} != {vertex.vertex_id}"
            )
    for edge in graph.edges:
        if edge.source not in vertex_by_id or edge.target not in vertex_by_id:
            raise ValueError(f"graph edge {edge.edge_id} references an unknown vertex")
    if dpgo_provenance_path is not None and Path(dpgo_provenance_path).is_file():
        provenance = json.loads(Path(dpgo_provenance_path).read_text())
        expected = provenance.get("source_sha256")
        if graph_path is not None and expected and sha256_file(graph_path) != expected:
            raise ValueError(
                f"verified graph checksum does not match DPGO provenance: {graph_path}"
            )


def _baseline(first: FrameKey, second: FrameKey, poses: dict[FrameKey, CbsPose]) -> float:
    return float(np.linalg.norm(poses[first].camera_center - poses[second].camera_center))


def _job_slug(kind: str, views: tuple[FrameKey, FrameKey], edge_id: int | None) -> str:
    if edge_id is not None:
        return f"loop_{edge_id:08d}_{views[0].robot_id}_{views[0].keyframe_id:08d}_{views[1].robot_id}_{views[1].keyframe_id:08d}"
    return f"{kind}_{views[0].robot_id}_{views[0].keyframe_id:08d}_{views[1].robot_id}_{views[1].keyframe_id:08d}"


def _nearest_baseline_context(
    view: FrameKey,
    ordered: Sequence[FrameKey],
    poses: dict[FrameKey, CbsPose],
    minimum_baseline: float,
) -> FrameKey:
    candidates = sorted(
        (candidate for candidate in ordered if candidate != view),
        key=lambda candidate: (abs(candidate.keyframe_id - view.keyframe_id), candidate.keyframe_id),
    )
    for candidate in candidates:
        if _baseline(view, candidate, poses) > minimum_baseline:
            return candidate
    raise ValueError(
        f"no temporal keyframe with baseline > {minimum_baseline:g} for {view.label()}"
    )


def build_two_view_jobs(
    artifacts: dict[str, TrackingArtifact],
    graph: Sim3PoseGraph,
    cbs_poses: dict[FrameKey, CbsPose],
    *,
    minimum_baseline: float = 1e-7,
    intra_pair_gap: int = 1,
    intra_pair_step: int = 2,
) -> list[PairJob]:
    """Build deterministic temporal intra pairs plus verified loops.

    The default gap=1, step=2 mode retains the original non-overlapping full
    keyframe coverage. Gap=2, step=2 selects the half-rate overlapping
    sequence (0,2), (2,4), ... for increased two-view parallax. If a selected
    pair is degenerate, each contributing view gets a nearest-temporal context.
    """

    if minimum_baseline < 0.0:
        raise ValueError("minimum baseline must be non-negative")
    if intra_pair_gap < 1 or intra_pair_step < 1:
        raise ValueError("intra pair gap and step must be positive")
    jobs: list[PairJob] = []
    for robot_id in sorted(artifacts, key=robot_sort_key):
        artifact = artifacts[robot_id]
        ordered = [
            FrameKey(robot_id, artifact.session_id, keyframe_id)
            for keyframe_id in range(artifact.keyframe_count)
        ]
        base_specs: list[tuple[tuple[FrameKey, FrameKey], tuple[bool, bool]]] = []
        if intra_pair_gap == 1 and intra_pair_step == 2:
            for first_index in range(0, len(ordered) - 1, 2):
                base_specs.append(
                    ((ordered[first_index], ordered[first_index + 1]), (True, True))
                )
            if len(ordered) % 2:
                if len(ordered) < 2:
                    raise ValueError(
                        f"{robot_id} has only one keyframe; two-view inference is impossible"
                    )
                base_specs.append(((ordered[-2], ordered[-1]), (False, True)))
            job_slug_kind = "intra"
        else:
            for first_index in range(
                0, len(ordered) - intra_pair_gap, intra_pair_step
            ):
                base_specs.append(
                    (
                        (
                            ordered[first_index],
                            ordered[first_index + intra_pair_gap],
                        ),
                        (True, True),
                    )
                )
            if not base_specs:
                raise ValueError(
                    f"{robot_id} has {len(ordered)} keyframes, fewer than required "
                    f"for intra pair gap {intra_pair_gap}"
                )
            job_slug_kind = f"intra_gap{intra_pair_gap}_step{intra_pair_step}"

        for views, mask in base_specs:
            if _baseline(*views, cbs_poses) > minimum_baseline:
                jobs.append(
                    PairJob(
                        _job_slug(job_slug_kind, views, None),
                        "intra_robot",
                        views,
                        mask,
                    )
                )
                continue
            for view, contributes in zip(views, mask, strict=True):
                if not contributes:
                    continue
                context = _nearest_baseline_context(view, ordered, cbs_poses, minimum_baseline)
                fallback_views = (view, context)
                jobs.append(
                    PairJob(
                        _job_slug(f"fallback_{job_slug_kind}", fallback_views, None),
                        "intra_robot_baseline_fallback",
                        fallback_views,
                        (True, False),
                        fallback_from=views,
                    )
                )

    vertex_by_id = {vertex.vertex_id: vertex for vertex in graph.vertices}
    seen_loop_pairs: set[tuple[FrameKey, FrameKey]] = set()
    loop_edges = sorted(
        (edge for edge in graph.edges if edge.edge_type == "inter_robot_loop_closure"),
        key=lambda edge: edge.edge_id,
    )
    for edge in loop_edges:
        views = (
            _frame_key_from_vertex(vertex_by_id[edge.source]),
            _frame_key_from_vertex(vertex_by_id[edge.target]),
        )
        canonical = tuple(sorted(views))
        if canonical in seen_loop_pairs:
            continue
        seen_loop_pairs.add(canonical)
        if _baseline(*views, cbs_poses) <= minimum_baseline:
            raise ValueError(
                f"verified loop edge {edge.edge_id} has effectively zero CBS baseline"
            )
        jobs.append(
            PairJob(
                _job_slug("loop", views, edge.edge_id),
                "inter_robot_loop_closure",
                views,
                (True, True),
                source_edge_id=edge.edge_id,
            )
        )
    return jobs


def frame_path_and_intrinsics(
    key: FrameKey, artifacts: dict[str, TrackingArtifact]
) -> tuple[Path, np.ndarray]:
    try:
        artifact = artifacts[key.robot_id]
    except KeyError as error:
        raise KeyError(f"no tracking artifact for {key.robot_id}") from error
    if artifact.session_id != key.session_id:
        raise ValueError(f"session mismatch for {key.label()}")
    intrinsics = recover_full_resolution_intrinsics(
        artifact.internal_intrinsics[key.keyframe_id], artifact.dpvo_resolution
    )
    return artifact.frame_path(key.keyframe_id), intrinsics


def pair_provenance(
    job: PairJob,
    artifacts: dict[str, TrackingArtifact],
    cbs_poses: dict[FrameKey, CbsPose],
    settings: InferenceSettings,
) -> dict[str, Any]:
    settings.validate()
    views = []
    for key in job.views:
        image_path, intrinsics = frame_path_and_intrinsics(key, artifacts)
        pose = cbs_poses[key]
        views.append(
            {
                **key.to_dict(),
                "image": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "intrinsics": intrinsics.tolist(),
                "rigid_world_to_camera": pose.camera_from_world.tolist(),
                "cbs_vertex_scale_diagnostic": pose.scale,
            }
        )
    value = {
        "cache_format": CACHE_FORMAT_NAME,
        "cache_version": CACHE_FORMAT_VERSION,
        "job": job.to_dict(),
        "views": views,
        "inference": settings.to_dict(),
    }
    value["fingerprint"] = _sha256_json(value)
    return value


def pair_cache_path(cache_dir: Path, job: PairJob) -> Path:
    return Path(cache_dir) / f"{job.job_id}.npz"


def save_pair_cache(
    path: Path,
    prediction: PairPrediction,
    job: PairJob,
    provenance: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    prediction = prediction.normalized()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            depth=prediction.depth.astype(np.float32),
            confidence=prediction.confidence.astype(np.float32),
            processed_rgb=prediction.processed_rgb.astype(np.uint8),
            intrinsics=prediction.intrinsics.astype(np.float32),
            da3_world_to_camera=prediction.extrinsics.astype(np.float32),
            contributing_views=np.asarray(job.contributing_views, dtype=np.bool_),
            provenance=np.asarray(_canonical_json(provenance)),
            pair_diagnostics=np.asarray(_canonical_json(diagnostics)),
        )
    temporary.replace(path)


def load_pair_cache(
    path: Path,
    expected_provenance: dict[str, Any] | None = None,
) -> tuple[PairPrediction, dict[str, Any], np.ndarray, dict[str, Any]] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            required = {
                "depth", "confidence", "processed_rgb", "intrinsics",
                "da3_world_to_camera", "contributing_views", "provenance",
                "pair_diagnostics",
            }
            if not required.issubset(cache.files):
                return None
            provenance = json.loads(str(cache["provenance"].item()))
            if expected_provenance is not None and provenance != expected_provenance:
                return None
            if (
                provenance.get("cache_format") != CACHE_FORMAT_NAME
                or provenance.get("cache_version") != CACHE_FORMAT_VERSION
                or provenance.get("inference", {}).get("inference_mode") != INFERENCE_MODE
            ):
                return None
            diagnostics = json.loads(str(cache["pair_diagnostics"].item()))
            required_diagnostics = {
                "cbs_baseline", "da3_baseline", "applied_depth_scale",
                "relative_rotation_error_degrees",
                "translation_direction_error_degrees",
                "cbs_source_sim3_scales", "invalid_da3_baseline_reason",
            }
            if not required_diagnostics.issubset(diagnostics):
                return None
            if diagnostics["invalid_da3_baseline_reason"] is not None:
                return None
            finite_diagnostics = [
                diagnostics["cbs_baseline"], diagnostics["da3_baseline"],
                diagnostics["applied_depth_scale"],
                diagnostics["relative_rotation_error_degrees"],
                diagnostics["translation_direction_error_degrees"],
                *diagnostics["cbs_source_sim3_scales"],
            ]
            if not np.isfinite(np.asarray(finite_diagnostics, dtype=np.float64)).all():
                return None
            mask = np.asarray(cache["contributing_views"], dtype=np.bool_)
            if mask.shape != (2,):
                return None
            prediction = PairPrediction(
                cache["depth"].copy(),
                cache["confidence"].copy(),
                cache["processed_rgb"].copy(),
                cache["intrinsics"].copy(),
                cache["da3_world_to_camera"].copy(),
            ).normalized()
            return prediction, provenance, mask, diagnostics
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


class Da3Backend:
    """Thin adapter around the pinned DA3 public API."""

    def __init__(self, settings: InferenceSettings, device: str = "cuda"):
        self.settings = settings
        self.device = device
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as error:
            raise RuntimeError(
                "Depth Anything 3 is unavailable. Run the Blackwell pixi environment "
                "or install the code revision declared in deploy/blackwell_ros2/pixi.toml."
            ) from error
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"DA3 device {device!r} was requested, but PyTorch reports no CUDA device"
            )
        installed_code_revision = None
        try:
            direct_url = importlib.metadata.distribution("depth-anything-3").read_text(
                "direct_url.json"
            )
            if direct_url:
                installed_code_revision = (
                    json.loads(direct_url).get("vcs_info", {}).get("commit_id")
                )
        except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
            pass
        if (
            installed_code_revision is not None
            and installed_code_revision != settings.code_revision
        ):
            raise RuntimeError(
                "installed Depth Anything 3 code revision does not match the requested "
                f"revision: {installed_code_revision} != {settings.code_revision}"
            )
        try:
            self.model = DepthAnything3.from_pretrained(
                settings.model, revision=settings.weights_revision
            ).to(device).eval()
        except Exception as error:
            raise RuntimeError(
                f"failed to load DA3 model {settings.model!r} at weights revision "
                f"{settings.weights_revision}: {error}"
            ) from error
        self._install_known_intrinsics_passthrough()
        try:
            package_version = importlib.metadata.version("depth-anything-3")
        except importlib.metadata.PackageNotFoundError:
            package_version = "unknown"
        self._revisions = {
            "model": settings.model,
            "requested_code_revision": settings.code_revision,
            "installed_code_revision": installed_code_revision,
            "requested_weights_revision": settings.weights_revision,
            "resolved_weights_revision": getattr(self.model, "_commit_hash", None),
            "package_version": package_version,
            "inference_mode": INFERENCE_MODE,
            "pose_conditioning": "no_input_extrinsics",
            "pose_estimation": "camera_decoder",
            "da3_output_extrinsics_convention": "world_to_camera",
            "da3_native_extrinsics_shape": "N x 3 x 4 (camera decoder; normalized to N x 4 x 4 in cache)",
            "da3_depth_pose_scale_relation": (
                "depth and predicted camera translations share the same arbitrary gauge"
            ),
            "known_intrinsics_adapter": "retain_preprocessed_input_intrinsics_without_pose_conditioning",
            "align_to_input_ext_scale": False,
        }

    @property
    def revisions(self) -> dict[str, Any]:
        return dict(self._revisions)

    def _install_known_intrinsics_passthrough(self) -> None:
        """Retain resized known K while keeping DA3 pose estimation unconditioned.

        At the pinned revision, DA3's camera decoder runs whenever extrinsics are
        absent, but ``_align_to_input_extrinsics_intrinsics`` also returns early
        and discards the resized input K.  This adapter only copies that already
        preprocessed K into the output; it never supplies or aligns extrinsics.
        """

        original = self.model._align_to_input_extrinsics_intrinsics

        def align(model, extrinsics, intrinsics, prediction, align_to_input_ext_scale=True):
            if extrinsics is not None:
                raise RuntimeError("corrected DA3 mode forbids input extrinsics")
            if align_to_input_ext_scale:
                raise RuntimeError("corrected DA3 mode forbids input-extrinsic scale alignment")
            prediction = original(extrinsics, intrinsics, prediction, False)
            if intrinsics is None:
                raise RuntimeError("corrected DA3 mode requires known intrinsics")
            prediction.intrinsics = (
                intrinsics.detach().cpu().numpy()
                if hasattr(intrinsics, "detach") else np.asarray(intrinsics)
            )
            return prediction

        self.model._align_to_input_extrinsics_intrinsics = types.MethodType(align, self.model)

    def infer(
        self,
        image_paths: Sequence[Path],
        intrinsics: np.ndarray,
        settings: InferenceSettings,
    ) -> PairPrediction:
        settings.validate()
        try:
            prediction = self.model.inference(
                image=[str(Path(path)) for path in image_paths],
                extrinsics=None,
                intrinsics=np.asarray(intrinsics, dtype=np.float32),
                align_to_input_ext_scale=False,
                infer_gs=False,
                use_ray_pose=False,
                process_res=settings.process_res,
                process_res_method=settings.process_res_method,
                export_dir=None,
            )
        except Exception as error:
            raise RuntimeError(
                "DA3 two-view inference failed. Check CUDA memory/support, image files, "
                f"and pinned model availability. Original error: {error}"
            ) from error
        if prediction.conf is None:
            raise RuntimeError("DA3 prediction has no confidence maps")
        if prediction.processed_images is None or prediction.intrinsics is None:
            raise RuntimeError("DA3 prediction omitted processed images or adjusted intrinsics")
        output_extrinsics = prediction.extrinsics
        if output_extrinsics is None:
            raise RuntimeError("DA3 prediction omitted camera-decoder poses")
        return PairPrediction(
            prediction.depth,
            prediction.conf,
            prediction.processed_images,
            prediction.intrinsics,
            output_extrinsics,
        ).normalized()


class LazyDa3Backend:
    """Load the large model only when at least one cache actually needs inference."""

    def __init__(self, settings: InferenceSettings, device: str = "cuda"):
        self.settings = settings
        self.device = device
        self._backend: Da3Backend | None = None

    def _get(self) -> Da3Backend:
        if self._backend is None:
            self._backend = Da3Backend(self.settings, self.device)
        return self._backend

    @property
    def revisions(self) -> dict[str, Any]:
        if self._backend is None:
            return {
                "requested_code_revision": self.settings.code_revision,
                "requested_weights_revision": self.settings.weights_revision,
                "model_loaded_this_run": False,
            }
        return {**self._backend.revisions, "model_loaded_this_run": True}

    def infer(
        self,
        image_paths: Sequence[Path],
        intrinsics: np.ndarray,
        settings: InferenceSettings,
    ) -> PairPrediction:
        return self._get().infer(image_paths, intrinsics, settings)


def pair_failure_path(cache_dir: Path, job: PairJob) -> Path:
    return Path(cache_dir).parent / "failures" / f"{job.job_id}.json"


def save_pair_failure(
    path: Path,
    job: PairJob,
    provenance: dict[str, Any],
    diagnostics: dict[str, Any],
    error: BaseException,
) -> None:
    """Atomically retain a rejected-pair diagnostic without creating a cache hit."""

    _atomic_json(
        path,
        {
            "cache_format": CACHE_FORMAT_NAME,
            "cache_version": CACHE_FORMAT_VERSION,
            "inference_mode": INFERENCE_MODE,
            "job": job.to_dict(),
            "provenance_fingerprint": provenance["fingerprint"],
            "diagnostics": diagnostics,
            "error": str(error),
        },
    )


def infer_jobs(
    jobs: Sequence[PairJob],
    artifacts: dict[str, TrackingArtifact],
    cbs_poses: dict[FrameKey, CbsPose],
    cache_dir: Path,
    settings: InferenceSettings,
    backend: InferenceBackend,
    *,
    force: bool = False,
) -> dict[str, int]:
    settings.validate()
    status = {
        "total": len(jobs), "reused": 0, "written": 0,
        "invalidated": 0, "rejected": 0,
    }
    for index, job in enumerate(jobs, start=1):
        provenance = pair_provenance(job, artifacts, cbs_poses, settings)
        cache_path = pair_cache_path(cache_dir, job)
        cached = None if force else load_pair_cache(cache_path, provenance)
        if cached is not None:
            status["reused"] += 1
            print(f"[DA3 {index}/{len(jobs)}] cache {job.job_id}", flush=True)
            continue
        if cache_path.exists():
            status["invalidated"] += 1
        paths_and_k = [frame_path_and_intrinsics(view, artifacts) for view in job.views]
        image_paths = [item[0] for item in paths_and_k]
        intrinsics = np.stack([item[1] for item in paths_and_k])
        print(f"[DA3 {index}/{len(jobs)}] infer {job.job_id}", flush=True)
        prediction = backend.infer(image_paths, intrinsics, settings)
        try:
            prediction, diagnostics = scale_prediction_to_cbs(
                prediction,
                [cbs_poses[view] for view in job.views],
                minimum_da3_baseline=settings.minimum_da3_baseline,
            )
        except InvalidDa3BaselineError as error:
            status["rejected"] += 1
            save_pair_failure(
                pair_failure_path(cache_dir, job),
                job,
                provenance,
                error.diagnostics,
                error,
            )
            raise RuntimeError(
                f"DA3 pair {job.job_id} has no usable pose baseline: {error}. "
                "Completed corrected caches remain resumable; no old pose-conditioned "
                "fallback was used."
            ) from error
        save_pair_cache(cache_path, prediction, job, provenance, diagnostics)
        status["written"] += 1
        print(
            f"[DA3 {index}/{len(jobs)}] scale={diagnostics['applied_depth_scale']:.8g} "
            f"CBS={diagnostics['cbs_baseline']:.8g} DA3={diagnostics['da3_baseline']:.8g}",
            flush=True,
        )
    return status


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    height, width = image.shape
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    du = u - u0
    dv = v - v0
    return (
        (1.0 - du) * (1.0 - dv) * image[v0, u0]
        + du * (1.0 - dv) * image[v0, u1]
        + (1.0 - du) * dv * image[v1, u0]
        + du * dv * image[v1, u1]
    )


@dataclass(frozen=True)
class DpvoPatchObservations:
    """Per-keyframe DPVO patch centres and depths in the stable local gauge."""

    u_full: np.ndarray
    v_full: np.ndarray
    stable_depth: np.ndarray


def recover_dpvo_patch_observations(
    artifact: TrackingArtifact,
) -> DpvoPatchObservations:
    """Recover full-image patch pixels and camera depths from a tracking artifact."""

    from scipy.spatial.transform import Rotation

    count = int(artifact.keyframe_count)
    patches = int(artifact.patches_per_keyframe)
    points = np.asarray(artifact.map_points, dtype=np.float64)
    poses = np.asarray(artifact.internal_poses_xyzw, dtype=np.float64)
    intrinsics = np.asarray(artifact.internal_intrinsics, dtype=np.float64)
    if points.shape != (count * patches, 3):
        raise ValueError(
            f"{artifact.robot_id} map points have shape {points.shape}, expected "
            f"{(count * patches, 3)}"
        )
    if poses.shape != (count, 7) or intrinsics.shape != (count, 4):
        raise ValueError(f"{artifact.robot_id} lacks per-keyframe DPVO pose/intrinsics")
    points = points.reshape(count, patches, 3)
    rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
    camera = np.einsum("nij,nmj->nmi", rotations, points) + poses[:, None, :3]
    depth = camera[:, :, 2]
    safe_depth = np.where(np.abs(depth) > 1e-15, depth, 1.0)
    u_internal = (
        intrinsics[:, None, 0] * camera[:, :, 0] / safe_depth
        + intrinsics[:, None, 2]
    )
    v_internal = (
        intrinsics[:, None, 1] * camera[:, :, 1] / safe_depth
        + intrinsics[:, None, 3]
    )
    transform = np.asarray(artifact.session_from_map, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{artifact.robot_id} has invalid session_from_map")
    stable_scale = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))
    if not np.isfinite(stable_scale) or stable_scale <= 0.0:
        raise ValueError(f"{artifact.robot_id} has invalid stable-map scale")
    resolution = int(artifact.dpvo_resolution)
    result = DpvoPatchObservations(
        (u_internal * resolution).astype(np.float64),
        (v_internal * resolution).astype(np.float64),
        (depth * stable_scale).astype(np.float64),
    )
    if not (
        np.isfinite(result.u_full).all()
        and np.isfinite(result.v_full).all()
        and np.isfinite(result.stable_depth).all()
    ):
        raise ValueError(f"{artifact.robot_id} has non-finite recovered DPVO patches")
    return result


def refine_prediction_scale_from_dpvo_keypoints(
    prediction: PairPrediction,
    job: PairJob,
    artifacts: dict[str, TrackingArtifact],
    cbs_poses: dict[FrameKey, CbsPose],
    observations: dict[str, DpvoPatchObservations],
    baseline_diagnostics: dict[str, Any],
    settings: FusionSettings,
) -> tuple[PairPrediction, dict[str, Any]]:
    """Robustly refine a pair's shared depth scale using DPVO patch depths.

    Stable robot-local patch depths are multiplied by the source CBS vertex
    Sim(3) scale to express them in the rigid CBS world gauge. One correction
    is then fitted in log-depth space across both views.
    """

    settings.validate()
    prediction = prediction.normalized()
    view_diagnostics = []
    pair_ratios = []
    for view_index, key in enumerate(job.views):
        artifact = artifacts[key.robot_id]
        patches = observations[key.robot_id]
        frame = key.keyframe_id
        full_intrinsics = recover_full_resolution_intrinsics(
            artifact.internal_intrinsics[frame], artifact.dpvo_resolution
        )
        count = patches.u_full.shape[1]
        pixels = np.stack(
            [patches.u_full[frame], patches.v_full[frame], np.ones(count)], axis=0
        )
        rays = np.linalg.solve(full_intrinsics, pixels)
        processed = prediction.intrinsics[view_index].astype(np.float64) @ rays
        u = processed[0] / processed[2]
        v = processed[1] / processed[2]
        height, width = prediction.depth[view_index].shape
        inside = (
            (u >= 0.0) & (v >= 0.0)
            & (u <= width - 1.0) & (v <= height - 1.0)
        )
        sampled_depth = np.full(count, np.nan, dtype=np.float64)
        sampled_confidence = np.full(count, np.nan, dtype=np.float64)
        if np.any(inside):
            sampled_depth[inside] = bilinear_sample(
                prediction.depth[view_index], u[inside], v[inside]
            )
            sampled_confidence[inside] = bilinear_sample(
                prediction.confidence[view_index], u[inside], v[inside]
            )
        target_depth = patches.stable_depth[frame] * cbs_poses[key].scale
        valid = (
            inside & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
            & np.isfinite(sampled_confidence) & np.isfinite(target_depth)
            & (target_depth >= settings.dpvo_keypoint_min_depth)
            & (target_depth <= settings.dpvo_keypoint_max_depth)
        )
        candidate_count = int(np.count_nonzero(valid))
        if candidate_count < settings.dpvo_keypoint_min_matches_per_view:
            raise RuntimeError(
                f"{job.job_id} view {view_index} has only {candidate_count} valid "
                "DPVO/DA3 depth matches"
            )
        confidence_cutoff = float(np.percentile(
            sampled_confidence[valid], settings.dpvo_keypoint_confidence_percentile
        ))
        valid &= sampled_confidence >= confidence_cutoff
        ratios = target_depth[valid] / sampled_depth[valid]
        if len(ratios) < settings.dpvo_keypoint_min_matches_per_view:
            raise RuntimeError(
                f"{job.job_id} view {view_index} has only {len(ratios)} "
                "high-confidence DPVO/DA3 depth matches"
            )
        pair_ratios.append(ratios)
        view_diagnostics.append({
            "frame": key.to_dict(),
            "candidate_matches": candidate_count,
            "high_confidence_matches": int(len(ratios)),
            "confidence_cutoff": confidence_cutoff,
            "median_depth_correction": float(np.median(ratios)),
            "cbs_source_sim3_scale": float(cbs_poses[key].scale),
        })

    ratios = np.concatenate(pair_ratios)
    log_ratios = np.log(ratios)
    log_median = float(np.median(log_ratios))
    log_mad = float(np.median(np.abs(log_ratios - log_median)))
    inlier_threshold = max(
        settings.dpvo_keypoint_mad_multiplier * 1.4826 * log_mad,
        float(np.log(1.05)),
    )
    inliers = np.abs(log_ratios - log_median) <= inlier_threshold
    minimum_pair_matches = 2 * settings.dpvo_keypoint_min_matches_per_view
    if int(np.count_nonzero(inliers)) < minimum_pair_matches:
        raise RuntimeError(
            f"{job.job_id} has too few robust DPVO/DA3 depth matches: "
            f"{int(np.count_nonzero(inliers))} < {minimum_pair_matches}"
        )
    correction = float(np.exp(np.median(log_ratios[inliers])))
    if not np.isfinite(correction) or correction <= 0.0:
        raise RuntimeError(f"{job.job_id} has invalid DPVO depth correction {correction}")
    baseline_scale = float(baseline_diagnostics["applied_depth_scale"])
    diagnostics = {
        **baseline_diagnostics,
        "scale_source": "dpvo_keypoint_depth_refinement",
        "baseline_applied_depth_scale": baseline_scale,
        "dpvo_keypoint_depth_correction": correction,
        "applied_depth_scale": baseline_scale * correction,
        "dpvo_keypoint_alignment": {
            "matches": int(len(ratios)),
            "inliers": int(np.count_nonzero(inliers)),
            "inlier_fraction": float(np.mean(inliers)),
            "log_ratio_mad": log_mad,
            "log_ratio_inlier_threshold": inlier_threshold,
            "views": view_diagnostics,
        },
    }
    refined = PairPrediction(
        prediction.depth * correction,
        prediction.confidence,
        prediction.processed_rgb,
        prediction.intrinsics,
        prediction.extrinsics,
    ).normalized()
    return refined, diagnostics


def backproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    camera = np.stack(
        (
            (np.asarray(u) - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (np.asarray(v) - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )
    camera_to_world = np.linalg.inv(np.asarray(world_to_camera, dtype=np.float64))
    return camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def reprojection_consistency_mask(
    points_world: np.ndarray,
    partner_depth: np.ndarray,
    partner_intrinsics: np.ndarray,
    partner_world_to_camera: np.ndarray,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Check depth agreement for points landing inside the partner image.

    Points outside the partner image are retained.  Points inside must project
    in front of the camera and agree with a positive finite bilinear depth.
    The returned second mask indicates which samples landed inside.
    """

    points = np.asarray(points_world, dtype=np.float64)
    ext = np.asarray(partner_world_to_camera, dtype=np.float64)
    camera = points @ ext[:3, :3].T + ext[:3, 3]
    z = camera[:, 2]
    k = np.asarray(partner_intrinsics, dtype=np.float64)
    safe_z = np.where(np.abs(z) > 1e-15, z, 1.0)
    u = k[0, 0] * camera[:, 0] / safe_z + k[0, 2]
    v = k[1, 1] * camera[:, 1] / safe_z + k[1, 2]
    height, width = np.asarray(partner_depth).shape
    inside = (
        (z > 0.0) & (u >= 0.0) & (v >= 0.0) &
        (u <= width - 1.0) & (v <= height - 1.0)
    )
    keep = np.ones(len(points), dtype=np.bool_)
    if np.any(inside):
        sampled = bilinear_sample(np.asarray(partner_depth), u[inside], v[inside])
        projected = z[inside]
        denominator = np.maximum(np.maximum(np.abs(projected), np.abs(sampled)), 1e-12)
        agreement = (
            np.isfinite(sampled) & (sampled > 0.0) &
            (np.abs(projected - sampled) / denominator <= relative_tolerance)
        )
        keep[inside] = agreement
    return keep, inside


@dataclass
class PointBatch:
    points: np.ndarray
    colors: np.ndarray
    weights: np.ndarray
    robot_ids: np.ndarray
    observations: np.ndarray

    @classmethod
    def empty(cls) -> "PointBatch":
        return cls(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )


def filter_prediction_view(
    prediction: PairPrediction,
    view_index: int,
    settings: FusionSettings,
    robot_index: int,
    cbs_world_to_camera: np.ndarray,
) -> tuple[PointBatch, dict[str, Any]]:
    settings.validate()
    prediction = prediction.normalized()
    cbs_world_to_camera = np.asarray(cbs_world_to_camera, dtype=np.float64)
    if cbs_world_to_camera.shape != (2, 4, 4) or not np.isfinite(cbs_world_to_camera).all():
        raise ValueError("fusion requires two finite rigid CBS world-to-camera poses")
    depth = prediction.depth[view_index]
    confidence = prediction.confidence[view_index]
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid_confidence = valid_depth & np.isfinite(confidence)
    if not np.any(valid_confidence):
        return PointBatch.empty(), {
            "positive_finite_depth": int(np.count_nonzero(valid_depth)),
            "after_confidence": 0,
            "after_far_depth": 0,
            "projected_inside_partner": 0,
            "reprojection_consistent_inside": 0,
            "reprojection_rejected_error": 0,
            "reprojection_rejected_outside": 0,
            "after_reprojection": 0,
            "after_stride": 0,
        }
    confidence_cutoff = float(
        np.percentile(confidence[valid_confidence], settings.confidence_percentile)
    )
    far_cutoff = float(np.percentile(depth[valid_depth], settings.far_depth_percentile))
    mask = valid_confidence & (confidence >= confidence_cutoff)
    after_confidence = int(np.count_nonzero(mask))
    mask &= depth <= far_cutoff
    after_far = int(np.count_nonzero(mask))
    yy, xx = np.nonzero(mask)
    points = backproject_pixels(
        xx,
        yy,
        depth[yy, xx],
        prediction.intrinsics[view_index],
        cbs_world_to_camera[view_index],
    )
    reprojection_keep, inside = reprojection_consistency_mask(
        points,
        prediction.depth[1 - view_index],
        prediction.intrinsics[1 - view_index],
        cbs_world_to_camera[1 - view_index],
        settings.reprojection_relative_tolerance,
    )
    consistent_inside = inside & reprojection_keep
    rejected_error = inside & ~reprojection_keep
    rejected_outside = ~inside if settings.require_reprojection_overlap else np.zeros_like(inside)
    keep = consistent_inside if settings.require_reprojection_overlap else reprojection_keep
    yy, xx, points = yy[keep], xx[keep], points[keep]
    stride_mask = (xx % settings.pixel_stride == 0) & (yy % settings.pixel_stride == 0)
    yy, xx, points = yy[stride_mask], xx[stride_mask], points[stride_mask]
    weights = confidence[yy, xx].astype(np.float64)
    weights = np.maximum(weights, np.finfo(np.float32).tiny)
    batch = PointBatch(
        points.astype(np.float32),
        prediction.processed_rgb[view_index, yy, xx].astype(np.uint8),
        weights.astype(np.float32),
        np.full(len(points), robot_index, dtype=np.int32),
        np.ones(len(points), dtype=np.int32),
    )
    return batch, {
        "positive_finite_depth": int(np.count_nonzero(valid_depth)),
        "confidence_cutoff": confidence_cutoff,
        "after_confidence": after_confidence,
        "far_depth_cutoff": far_cutoff,
        "after_far_depth": after_far,
        "projected_inside_partner": int(np.count_nonzero(inside)),
        "reprojection_consistent_inside": int(np.count_nonzero(consistent_inside)),
        "reprojection_rejected_error": int(np.count_nonzero(rejected_error)),
        "reprojection_rejected_outside": int(np.count_nonzero(rejected_outside)),
        "after_reprojection": int(np.count_nonzero(keep)),
        "after_stride": len(points),
    }


def concatenate_batches(batches: Iterable[PointBatch]) -> PointBatch:
    batches = [batch for batch in batches if len(batch.points)]
    if not batches:
        return PointBatch.empty()
    return PointBatch(
        np.concatenate([batch.points for batch in batches]),
        np.concatenate([batch.colors for batch in batches]),
        np.concatenate([batch.weights for batch in batches]),
        np.concatenate([batch.robot_ids for batch in batches]),
        np.concatenate([batch.observations for batch in batches]),
    )


def voxel_fuse(batch: PointBatch, voxel_size: float) -> PointBatch:
    if len(batch.points) == 0:
        return PointBatch.empty()
    if voxel_size <= 0.0 or not np.isfinite(voxel_size):
        raise ValueError("voxel size must be finite and positive")
    finite = (
        np.isfinite(batch.points).all(axis=1) & np.isfinite(batch.weights) &
        (batch.weights > 0.0)
    )
    if not np.any(finite):
        return PointBatch.empty()
    points = np.asarray(batch.points[finite], dtype=np.float64)
    colors = np.asarray(batch.colors[finite], dtype=np.float64)
    weights = np.asarray(batch.weights[finite], dtype=np.float64)
    robots = np.asarray(batch.robot_ids[finite], dtype=np.int32)
    observations = np.asarray(batch.observations[finite], dtype=np.int64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    weight_sum = np.bincount(inverse, weights=weights, minlength=count)
    fused_points = np.stack(
        [np.bincount(inverse, weights=weights * points[:, axis], minlength=count) for axis in range(3)],
        axis=1,
    ) / weight_sum[:, None]
    fused_colors = np.stack(
        [np.bincount(inverse, weights=weights * colors[:, axis], minlength=count) for axis in range(3)],
        axis=1,
    ) / weight_sum[:, None]
    fused_observations = np.bincount(inverse, weights=observations, minlength=count)
    unique_robots = np.unique(robots)
    ownership_weight = np.stack(
        [np.bincount(inverse, weights=weights * (robots == robot), minlength=count) for robot in unique_robots],
        axis=1,
    )
    fused_robots = unique_robots[np.argmax(ownership_weight, axis=1)]
    return PointBatch(
        fused_points.astype(np.float32),
        np.clip(np.rint(fused_colors), 0, 255).astype(np.uint8),
        weight_sum.astype(np.float32),
        fused_robots.astype(np.int32),
        fused_observations.astype(np.int32),
    )


def fuse_cached_jobs(
    jobs: Sequence[PairJob],
    artifacts: dict[str, TrackingArtifact],
    cbs_poses: dict[FrameKey, CbsPose],
    cache_dir: Path,
    inference_settings: InferenceSettings,
    fusion_settings: FusionSettings,
) -> tuple[PointBatch, dict[str, Any]]:
    robot_ids = sorted(artifacts, key=robot_sort_key)
    robot_index = {robot_id: index for index, robot_id in enumerate(robot_ids)}
    raw_by_robot: dict[str, list[PointBatch]] = defaultdict(list)
    pair_stats: dict[str, Any] = {}
    contribution_count = {robot_id: 0 for robot_id in robot_ids}
    dpvo_observations = (
        {
            robot_id: recover_dpvo_patch_observations(artifact)
            for robot_id, artifact in artifacts.items()
        }
        if fusion_settings.depth_scale_refinement == "dpvo_keypoints"
        else {}
    )
    for job in jobs:
        provenance = pair_provenance(job, artifacts, cbs_poses, inference_settings)
        cache_path = pair_cache_path(cache_dir, job)
        cached = load_pair_cache(cache_path, provenance)
        if cached is None:
            raise RuntimeError(
                f"missing, corrupt, or stale pair cache for {job.job_id}: {cache_path}. "
                "Run --stage infer (or --stage all) first."
            )
        prediction, _, cached_mask, diagnostics = cached
        if fusion_settings.depth_scale_refinement == "dpvo_keypoints":
            prediction, diagnostics = refine_prediction_scale_from_dpvo_keypoints(
                prediction,
                job,
                artifacts,
                cbs_poses,
                dpvo_observations,
                diagnostics,
                fusion_settings,
            )
        expected_mask = np.asarray(job.contributing_views, dtype=np.bool_)
        if not np.array_equal(cached_mask, expected_mask):
            raise RuntimeError(f"contributing-view mask mismatch in {cache_path}")
        cbs_world_to_camera = np.stack(
            [cbs_poses[view].camera_from_world for view in job.views]
        )
        view_stats = []
        for view_index, contributes in enumerate(job.contributing_views):
            if not contributes:
                view_stats.append({"contributes": False})
                continue
            key = job.views[view_index]
            batch, stats = filter_prediction_view(
                prediction,
                view_index,
                fusion_settings,
                robot_index[key.robot_id],
                cbs_world_to_camera,
            )
            raw_by_robot[key.robot_id].append(batch)
            contribution_count[key.robot_id] += 1
            view_stats.append({"contributes": True, "frame": key.to_dict(), **stats})
        pair_stats[job.job_id] = {
            "views": view_stats,
            "scale_and_relative_pose": diagnostics,
        }

    per_robot: list[PointBatch] = []
    robot_stats = {}
    for robot_id in robot_ids:
        raw = concatenate_batches(raw_by_robot[robot_id])
        fused = voxel_fuse(raw, fusion_settings.voxel_size)
        if contribution_count[robot_id] and len(fused.points) == 0:
            raise RuntimeError(f"filtering produced an empty dense cloud for {robot_id}")
        per_robot.append(fused)
        robot_stats[robot_id] = {
            "contributing_views": contribution_count[robot_id],
            "points_before_voxel_fusion": len(raw.points),
            "points_after_robot_voxel_fusion": len(fused.points),
        }
    global_input = concatenate_batches(per_robot)
    global_cloud = voxel_fuse(global_input, fusion_settings.voxel_size)
    if not len(global_cloud.points):
        raise RuntimeError("filtering and fusion produced an empty dense cloud")
    return global_cloud, {
        "pairs": pair_stats,
        "robots": robot_stats,
        "points_after_global_voxel_fusion": len(global_cloud.points),
    }


def write_dense_ply(path: Path, cloud: PointBatch, robot_names: Sequence[str]) -> None:
    from plyfile import PlyData, PlyElement

    if not len(cloud.points):
        raise ValueError("refusing to write an empty dense PLY")
    block = np.empty(
        len(cloud.points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("robot_id", "u1"), ("confidence_weight", "<f4"),
            ("observation_count", "<i4"),
        ],
    )
    block["x"], block["y"], block["z"] = cloud.points.T
    block["red"], block["green"], block["blue"] = cloud.colors.T
    if len(robot_names) > 256 or np.any(cloud.robot_ids > 255):
        raise ValueError("PLY robot ownership exceeds uint8 range")
    block["robot_id"] = cloud.robot_ids.astype(np.uint8)
    block["confidence_weight"] = cloud.weights
    block["observation_count"] = cloud.observations
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    PlyData([PlyElement.describe(block, "vertex")], text=False).write(temporary)
    temporary.replace(path)


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    points = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=1).astype(np.float32)
    colors = None
    if {"red", "green", "blue"}.issubset(vertex.dtype.names or ()):
        colors = np.stack([vertex[channel] for channel in ("red", "green", "blue")], axis=1).astype(np.uint8)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite] if colors is not None else None


def sparse_to_dense_diagnostics(sparse_path: Path | None, dense: np.ndarray) -> dict[str, Any]:
    if sparse_path is None or not Path(sparse_path).is_file():
        return {"available": False, "reason": "sparse PLY not found"}
    sparse, _ = read_ply_points(Path(sparse_path))
    if not len(sparse):
        return {"available": False, "reason": "sparse PLY contains no finite points"}
    distances, _ = cKDTree(np.asarray(dense, dtype=np.float64)).query(sparse, k=1)
    quantiles = np.percentile(distances, [0, 25, 50, 75, 90, 95, 99, 100])
    return {
        "available": True,
        "sparse_point_count": len(sparse),
        "dense_point_count": len(dense),
        "distance_cbs_units": {
            "min": float(quantiles[0]), "p25": float(quantiles[1]),
            "median": float(quantiles[2]), "p75": float(quantiles[3]),
            "p90": float(quantiles[4]), "p95": float(quantiles[5]),
            "p99": float(quantiles[6]), "max": float(quantiles[7]),
            "mean": float(np.mean(distances)),
        },
    }


ROBOT_COLORS = np.asarray(
    [
        [0, 170, 255], [255, 95, 85], [110, 220, 120], [220, 50, 40],
        [170, 68, 153], [221, 204, 119], [68, 170, 153], [136, 204, 238],
        [51, 34, 136], [238, 51, 119], [153, 153, 51], [102, 17, 0],
        [17, 119, 51],
    ],
    dtype=np.uint8,
)


def dense_rrd_recording_id(path: Path) -> str:
    """Return a stable recording ID unique to this dense output path."""

    output_identity = str(Path(path).expanduser().resolve())
    digest = hashlib.sha256(output_identity.encode("utf-8")).hexdigest()[:16]
    return f"dpvo-cbs-da3-dense-map-{digest}"


def write_dense_rrd(
    path: Path,
    cloud: PointBatch,
    robot_names: Sequence[str],
    graph: Sim3PoseGraph,
    cbs_poses: dict[FrameKey, CbsPose],
    sparse_path: Path | None = None,
) -> str:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError("rerun-sdk is required to export the dense RRD") from error

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="CBS DA3 dense RGB",
                origin="world",
                contents=["world/dense/rgb/**", "world/cbs/**", "world/sparse/**"],
            ),
            rrb.Spatial3DView(
                name="Dense robot ownership",
                origin="world",
                contents=["world/dense/by_robot/**", "world/cbs/**"],
            ),
        ),
        collapse_panels=True,
    )
    path = Path(path)
    recording_id = dense_rrd_recording_id(path)
    rr.init(
        "DPVO CBS DA3 Dense Map",
        recording_id=recording_id,
        default_blueprint=blueprint,
        strict=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(path)
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log("world/dense/rgb/points", rr.Points3D(cloud.points, colors=cloud.colors, radii=rr.Radius.ui_points(1.0)), static=True)
    ownership_colors = ROBOT_COLORS[cloud.robot_ids % len(ROBOT_COLORS)]
    rr.log("world/dense/by_robot/points", rr.Points3D(cloud.points, colors=ownership_colors, radii=rr.Radius.ui_points(1.0)), static=True)
    if sparse_path is not None and Path(sparse_path).is_file():
        sparse_points, sparse_colors = read_ply_points(Path(sparse_path))
        rr.log(
            "world/sparse/existing_map",
            rr.Points3D(
                sparse_points,
                colors=sparse_colors if sparse_colors is not None else [150, 150, 150],
                radii=rr.Radius.ui_points(0.75),
            ),
            static=True,
        )
    vertex_by_id = {vertex.vertex_id: vertex for vertex in graph.vertices}
    for robot_index, robot_id in enumerate(robot_names):
        keys = sorted(
            (key for key in cbs_poses if key.robot_id == robot_id),
            key=lambda key: key.keyframe_id,
        )
        trajectory = np.asarray([cbs_poses[key].camera_center for key in keys], dtype=np.float32)
        color = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        rr.log(f"world/cbs/trajectories/{robot_id}", rr.LineStrips3D([trajectory], colors=color, radii=rr.Radius.ui_points(2.0)), static=True)
        rr.log(f"world/cbs/keyframes/{robot_id}", rr.Points3D(trajectory, colors=color, radii=rr.Radius.ui_points(1.5)), static=True)
    loops = []
    labels = []
    for edge in graph.edges:
        if edge.edge_type != "inter_robot_loop_closure":
            continue
        source = _frame_key_from_vertex(vertex_by_id[edge.source])
        target = _frame_key_from_vertex(vertex_by_id[edge.target])
        loops.append(np.asarray([cbs_poses[source].camera_center, cbs_poses[target].camera_center], dtype=np.float32))
        labels.append(f"{source.robot_id}:{source.keyframe_id} -> {target.robot_id}:{target.keyframe_id}")
    if loops:
        rr.log("world/cbs/inter_robot_loops", rr.LineStrips3D(loops, colors=[255, 190, 50], labels=labels, radii=rr.Radius.ui_points(2.5)), static=True)
    rr.disconnect()
    return recording_id


@dataclass(frozen=True)
class ReconstructionPaths:
    run_dir: Path
    tracking_dir: Path
    graph_path: Path
    cbs_csv: Path
    sparse_ply: Path | None
    output_dir: Path

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        *,
        tracking_dir: Path | None = None,
        graph_path: Path | None = None,
        cbs_csv: Path | None = None,
        sparse_ply: Path | None = None,
        output_dir: Path | None = None,
    ) -> "ReconstructionPaths":
        run_dir = Path(run_dir).expanduser().resolve()
        default_sparse = run_dir / "plots" / "cbs_posewise_joint_map.ply"
        return cls(
            run_dir,
            Path(tracking_dir).expanduser().resolve() if tracking_dir else run_dir / "tracking",
            Path(graph_path).expanduser().resolve() if graph_path else run_dir / "geometric_verification" / "unoptimized_verified_graph.json",
            Path(cbs_csv).expanduser().resolve() if cbs_csv else run_dir / "dpgo" / "cbs.csv",
            Path(sparse_ply).expanduser().resolve() if sparse_ply else (default_sparse if default_sparse.is_file() else None),
            Path(output_dir).expanduser().resolve()
            if output_dir
            else run_dir / "dense_reconstruction" / "da3_two_view_pose_scaled",
        )


@dataclass
class LoadedReconstruction:
    paths: ReconstructionPaths
    artifacts: dict[str, TrackingArtifact]
    graph: Sim3PoseGraph
    cbs_poses: dict[FrameKey, CbsPose]
    jobs: list[PairJob]
    input_checksums: dict[str, Any] = field(default_factory=dict)
    intra_pair_gap: int = 1
    intra_pair_step: int = 2


def load_reconstruction(
    paths: ReconstructionPaths,
    minimum_baseline: float = 1e-7,
    intra_pair_gap: int = 1,
    intra_pair_step: int = 2,
) -> LoadedReconstruction:
    for required in (paths.tracking_dir, paths.graph_path, paths.cbs_csv):
        if not required.exists():
            raise FileNotFoundError(f"required reconstruction input does not exist: {required}")
    artifacts = load_artifact_root(paths.tracking_dir)
    graph = read_json(paths.graph_path)
    cbs_poses, cbs_hash = load_cbs_csv(paths.cbs_csv)
    provenance_path = paths.run_dir / "dpgo" / "offline_dpgo_provenance.json"
    validate_inputs(
        artifacts,
        graph,
        cbs_poses,
        graph_path=paths.graph_path,
        dpgo_provenance_path=provenance_path,
    )
    jobs = build_two_view_jobs(
        artifacts,
        graph,
        cbs_poses,
        minimum_baseline=minimum_baseline,
        intra_pair_gap=intra_pair_gap,
        intra_pair_step=intra_pair_step,
    )
    return LoadedReconstruction(
        paths,
        artifacts,
        graph,
        cbs_poses,
        jobs,
        {
            "verified_graph_sha256": sha256_file(paths.graph_path),
            "cbs_csv_sha256": cbs_hash,
            "tracking": {
                robot_id: {
                    "state_sha256": artifact.manifest["state_sha256"],
                    "map_sha256": artifact.manifest["map_sha256"],
                    "keyframe_images_sha256": artifact.manifest["keyframe_images_sha256"],
                }
                for robot_id, artifact in sorted(artifacts.items(), key=lambda item: robot_sort_key(item[0]))
            },
        },
        intra_pair_gap,
        intra_pair_step,
    )


def build_manifest(
    loaded: LoadedReconstruction,
    selected_jobs: Sequence[PairJob],
    inference_settings: InferenceSettings,
    fusion_settings: FusionSettings,
    cache_status: dict[str, int],
    cloud: PointBatch,
    fusion_stats: dict[str, Any],
    model_revisions: dict[str, Any],
) -> dict[str, Any]:
    paths = loaded.paths
    base_jobs = [job for job in loaded.jobs if job.kind.startswith("intra_robot")]
    loop_jobs = [job for job in loaded.jobs if job.kind == "inter_robot_loop_closure"]
    selected_ids = {job.job_id for job in selected_jobs}
    sparse_diagnostics = sparse_to_dense_diagnostics(paths.sparse_ply, cloud.points)
    robot_names = sorted(loaded.artifacts, key=robot_sort_key)
    finite_by_robot = {
        robot_id: int(np.count_nonzero(cloud.robot_ids == index))
        for index, robot_id in enumerate(robot_names)
    }
    pair_diagnostics = [
        value["scale_and_relative_pose"]
        for value in fusion_stats["pairs"].values()
    ]
    scales = np.asarray(
        [value["applied_depth_scale"] for value in pair_diagnostics], dtype=np.float64
    )
    cbs_baselines = np.asarray(
        [value["cbs_baseline"] for value in pair_diagnostics], dtype=np.float64
    )
    da3_baselines = np.asarray(
        [value["da3_baseline"] for value in pair_diagnostics], dtype=np.float64
    )
    rotation_errors = np.asarray(
        [value["relative_rotation_error_degrees"] for value in pair_diagnostics],
        dtype=np.float64,
    )
    direction_errors = np.asarray(
        [value["translation_direction_error_degrees"] for value in pair_diagnostics],
        dtype=np.float64,
    )

    def distribution(values: np.ndarray) -> dict[str, Any]:
        if not len(values):
            return {"count": 0}
        if not np.isfinite(values).all():
            raise RuntimeError("refusing to manifest non-finite pair-scale diagnostics")
        return {
            "count": len(values),
            "minimum": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values)),
        }
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "coordinate_frame": "CBS robot0 frame from per-robot relative anchor estimates (not asserted metric)",
        "inputs": {
            "run_dir": str(paths.run_dir),
            "tracking_dir": str(paths.tracking_dir),
            "verified_graph": str(paths.graph_path),
            "cbs_csv": str(paths.cbs_csv),
            "sparse_ply": str(paths.sparse_ply) if paths.sparse_ply else None,
            "checksums": loaded.input_checksums,
        },
        "outputs": {
            "ply": str(paths.output_dir / "cbs_da3_dense_map.ply"),
            "rrd": str(paths.output_dir / "cbs_da3_dense_map.rrd"),
            "manifest": str(paths.output_dir / "cbs_da3_dense_map.json"),
            "pair_cache": str(paths.output_dir / "cache" / "pairs"),
        },
        "model": {**inference_settings.to_dict(), **model_revisions},
        "scale_handling": {
            "mode": INFERENCE_MODE,
            "cbs_pose_projection": (
                "Sim(3) camera-to-world projected to rigid SE(3), preserving camera centre; "
                "source scale retained only as diagnostics"
            ),
            "da3_pose_source": "camera decoder with no input extrinsics",
            "depth_conversion": (
                "DA3/CBS baseline scale refined by robust DPVO keypoint depth alignment"
                if fusion_settings.depth_scale_refinement == "dpvo_keypoints"
                else "CBS camera baseline divided by DA3 predicted camera baseline"
            ),
            "depth_scale_refinement": fusion_settings.depth_scale_refinement,
            "backprojection_pose_source": "rigid CBS world-to-camera poses",
            "pair_diagnostics": {
                "accepted": len(pair_diagnostics),
                "rejected": int(cache_status.get("rejected", 0)),
                "applied_depth_scale": distribution(scales),
                "cbs_baseline": distribution(cbs_baselines),
                "da3_baseline": distribution(da3_baselines),
                "relative_rotation_error_degrees": distribution(rotation_errors),
                "translation_direction_error_degrees": distribution(direction_errors),
            },
        },
        "filters": fusion_settings.to_dict(),
        "inventory": {
            "total_keyframes": sum(artifact.keyframe_count for artifact in loaded.artifacts.values()),
            "temporal_sampling": {
                "intra_pair_gap": loaded.intra_pair_gap,
                "intra_pair_step": loaded.intra_pair_step,
                "description": (
                    f"(0,{loaded.intra_pair_gap}), "
                    f"({loaded.intra_pair_step},"
                    f"{loaded.intra_pair_step + loaded.intra_pair_gap}), ..."
                ),
            },
            "all_base_pairs": len(base_jobs),
            "all_verified_loop_pairs": len(loop_jobs),
            "all_pairs": len(loaded.jobs),
            "selected_pairs": len(selected_jobs),
            "selection_complete": len(selected_jobs) == len(loaded.jobs),
            "jobs": [
                {**job.to_dict(), "selected": job.job_id in selected_ids}
                for job in loaded.jobs
            ],
        },
        "cache": cache_status,
        "point_counts": {
            "dense_global": len(cloud.points),
            "dense_by_dominant_robot": finite_by_robot,
            "sparse_existing": sparse_diagnostics.get("sparse_point_count"),
            "dense_exceeds_sparse": (
                len(cloud.points) > sparse_diagnostics["sparse_point_count"]
                if sparse_diagnostics.get("available") else None
            ),
            "robot_id_encoding": {
                str(index): robot_id for index, robot_id in enumerate(robot_names)
            },
        },
        "fusion": fusion_stats,
        "sparse_to_dense_nearest_neighbor": sparse_diagnostics,
    }


def export_reconstruction(
    loaded: LoadedReconstruction,
    selected_jobs: Sequence[PairJob],
    inference_settings: InferenceSettings,
    fusion_settings: FusionSettings,
    cache_status: dict[str, int] | None = None,
    model_revisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = loaded.paths.output_dir
    cache_dir = output_dir / "cache" / "pairs"
    cloud, fusion_stats = fuse_cached_jobs(
        selected_jobs,
        loaded.artifacts,
        loaded.cbs_poses,
        cache_dir,
        inference_settings,
        fusion_settings,
    )
    robot_names = sorted(loaded.artifacts, key=robot_sort_key)
    write_dense_ply(output_dir / "cbs_da3_dense_map.ply", cloud, robot_names)
    rrd_recording_id = write_dense_rrd(
        output_dir / "cbs_da3_dense_map.rrd",
        cloud,
        robot_names,
        loaded.graph,
        loaded.cbs_poses,
        loaded.paths.sparse_ply,
    )
    manifest = build_manifest(
        loaded,
        selected_jobs,
        inference_settings,
        fusion_settings,
        cache_status
        or {
            "total": len(selected_jobs), "reused": len(selected_jobs),
            "written": 0, "invalidated": 0, "rejected": 0,
        },
        cloud,
        fusion_stats,
        model_revisions or {},
    )
    manifest["outputs"]["rrd_recording_id"] = rrd_recording_id
    _atomic_json(output_dir / "cbs_da3_dense_map.json", manifest)
    return manifest

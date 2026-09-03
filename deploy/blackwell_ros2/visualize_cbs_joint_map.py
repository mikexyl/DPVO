#!/usr/bin/env python3
"""Build joint sparse maps from DPVO points and Sim(3) pose-graph solutions.

The DPVO recording stores each robot's final sparse points in the mutable
internal map chart, together with the final ``session_from_map`` similarity.
The unoptimized graph stores camera poses in the stable session chart, while
the CBS CSV stores the distributed trajectory in robot0's frame. Two map
visualization modes are supported:

``map-level`` preserves the complete local DPVO map and estimates one Sim(3)
from a regenerated replay to the saved trajectory. This matches the online
ROS/Rerun visualization, which places each robot map below one parent Sim(3).

``posewise`` applies the per-keyframe deformation

    p_global = Y_solution[k] * inverse(X_source[k])
               * T_session_map * p_internal.

Pose-wise deformation is exact only when points and poses came from the same
DPVO run. It is retained as a diagnostic for exact saved Rerun recordings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import rerun as rr
import rerun.blueprint as rrb
import rerun.dataframe as rdf
from scipy.spatial.transform import Rotation

from dpvo.loop_closure.centralized import Sim3
from dpvo.loop_closure.pose_graph import read_json, sim3_from_dict, sim3_to_dict


ROBOT_COLORS = {
    "robot0": np.array([0, 170, 255], dtype=np.uint8),
    "robot1": np.array([255, 95, 85], dtype=np.uint8),
    "robot2": np.array([110, 220, 120], dtype=np.uint8),
    "robot3": np.array([220, 50, 40], dtype=np.uint8),
    "robot4": np.array([170, 68, 153], dtype=np.uint8),
    "robot5": np.array([221, 204, 119], dtype=np.uint8),
    "robot6": np.array([68, 170, 153], dtype=np.uint8),
    "robot7": np.array([136, 204, 238], dtype=np.uint8),
    "robot8": np.array([51, 34, 136], dtype=np.uint8),
    "robot9": np.array([238, 51, 119], dtype=np.uint8),
    "robot10": np.array([153, 153, 51], dtype=np.uint8),
    "robot11": np.array([102, 17, 0], dtype=np.uint8),
    "robot12": np.array([17, 119, 51], dtype=np.uint8),
    "robot13": np.array([0, 102, 102], dtype=np.uint8),
}


def _robot_sort_key(robot_id: str):
    suffix = robot_id.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot_id)


def _robot_index(robot_id: str) -> int:
    suffix = robot_id.removeprefix("robot")
    if not suffix.isdigit():
        raise ValueError(f"PLY export requires a numeric robot id: {robot_id}")
    index = int(suffix)
    if not 0 <= index <= np.iinfo(np.uint8).max:
        raise ValueError(f"robot id is outside uint8 PLY range: {robot_id}")
    return index


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Warp final DPVO sparse points with a CBS Sim(3) solution."
    )
    point_source = parser.add_mutually_exclusive_group(required=True)
    point_source.add_argument(
        "--input-rrd",
        type=Path,
        help="Multi-robot Rerun recording containing final DPVO point maps.",
    )
    parser.add_argument(
        "--input-recording-id",
        help=(
            "Recording ID to select when --input-rrd contains more than one "
            "recording. Required for multi-recording RRD archives."
        ),
    )
    point_source.add_argument(
        "--input-npz",
        action="append",
        metavar="ROBOT=PATH",
        help=(
            "Per-robot NPZ exported by run_euroc_bag.py. Repeat once per "
            "robot, for example --input-npz robot0=mh01.npz."
        ),
    )
    parser.add_argument("--input-graph", type=Path, required=True)
    parser.add_argument("--cbs-csv", type=Path, required=True)
    parser.add_argument(
        "--primary-label",
        default="CBS",
        help="Display label for --cbs-csv and the solution exported to PLY.",
    )
    parser.add_argument(
        "--comparison-csv",
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Optional additional pose solution to visualize beside CBS. "
            "Repeat for more than one solution."
        ),
    )
    parser.add_argument(
        "--online-map-transforms-json",
        type=Path,
        help=(
            "Optional online centralized result JSON. Its per-robot map "
            "transforms are applied to the unoptimized graph and shown as "
            "'Online centralized'."
        ),
    )
    parser.add_argument("--output-rrd", type=Path, required=True)
    parser.add_argument(
        "--output-ply",
        type=Path,
        help="Optional binary PLY containing the flattened joint point map.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help="Optional JSON manifest; defaults beside --output-rrd.",
    )
    parser.add_argument(
        "--recording-id",
        default="dpvo-cbs-joint-map",
        help="Rerun recording ID for the generated recording.",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=1.5,
        help="Point radius in Rerun UI points.",
    )
    parser.add_argument(
        "--host-distance-quantile",
        type=float,
        default=0.995,
        help=(
            "Per-robot quantile used to reject extreme DPVO depth outliers "
            "before visualization; use 1.0 to retain all finite points."
        ),
    )
    parser.add_argument(
        "--warp-mode",
        choices=("map-level", "posewise"),
        default="map-level",
        help=(
            "Use one similarity per robot (online-compatible, default), or "
            "deform every keyframe point block independently."
        ),
    )
    return parser.parse_args()


def _read_cbs_csv(path: Path) -> dict[tuple[str, str, int], Sim3]:
    estimates = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["robot_id"],
                row["session_id"],
                int(row["keyframe_id"]),
            )
            if key in estimates:
                raise ValueError(f"duplicate CBS estimate for {key}")
            scale = float(row["scale"])
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"invalid CBS scale for {key}: {scale}")
            estimates[key] = Sim3(
                [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                Rotation.from_quat(
                    [
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                        float(row["qw"]),
                    ]
                ).as_matrix(),
                scale,
            )
    if not estimates:
        raise ValueError(f"CBS CSV contains no estimates: {path}")
    return estimates


def _parse_npz_sources(values) -> dict[str, Path]:
    sources = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"expected ROBOT=PATH for --input-npz, got {value!r}")
        robot_id, path = value.split("=", 1)
        robot_id = robot_id.strip()
        if not robot_id:
            raise ValueError(f"empty robot ID in --input-npz {value!r}")
        if robot_id in sources:
            raise ValueError(f"duplicate --input-npz for {robot_id}")
        sources[robot_id] = Path(path).expanduser()
    return sources


def _parse_labeled_paths(
    values, reserved_labels: set[str] | None = None
) -> dict[str, Path]:
    sources = {}
    reserved_labels = reserved_labels or set()
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"expected LABEL=PATH for --comparison-csv, got {value!r}"
            )
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty label in --comparison-csv {value!r}")
        if label in reserved_labels or label in sources:
            raise ValueError(f"duplicate solution label {label!r}")
        sources[label] = Path(path).expanduser()
    return sources


def _load_input_recording(path: Path, recording_id: str | None):
    archive = rdf.load_archive(path)
    recordings = archive.all_recordings()
    if recording_id is not None:
        matches = [
            recording
            for recording in recordings
            if str(recording.recording_id()) == recording_id
        ]
        if len(matches) != 1:
            available = [str(recording.recording_id()) for recording in recordings]
            raise ValueError(
                f"RRD contains {len(matches)} recordings named {recording_id!r}; "
                f"available recording IDs: {available}"
            )
        return matches[0]
    if len(recordings) != 1:
        available = [str(recording.recording_id()) for recording in recordings]
        raise ValueError(
            "--input-recording-id is required because the RRD contains "
            f"{len(recordings)} recordings: {available}"
        )
    return recordings[0]


def _read_online_map_estimates(path: Path, graph) -> dict:
    data = json.loads(path.read_text())
    transforms = data.get("robots")
    if transforms is None and "map_transforms" in data:
        transforms = {
            key: value["online_map_centralized"]
            for key, value in data["map_transforms"].items()
        }
    if not isinstance(transforms, dict):
        raise ValueError(
            f"online map transform JSON has no 'robots' mapping: {path}"
        )

    estimates = {}
    for vertex in graph.vertices:
        robot_key = f"{vertex.robot_id}:{vertex.session_id}"
        if robot_key not in transforms:
            raise ValueError(f"online map JSON is missing {robot_key}")
        map_transform = sim3_from_dict(transforms[robot_key])
        estimates[(vertex.robot_id, vertex.session_id, vertex.keyframe_id)] = (
            map_transform.compose(vertex.estimate)
        )
    return estimates


def _last_points_frame(recording: rdf.Recording, robot_id: str) -> int:
    entity = f"world/robots/{robot_id}/map/points"
    indicator = f"/{entity}:Points3DIndicator"
    view = recording.view(
        index="frame",
        contents=f"{entity}/**",
        include_indicator_columns=True,
    )
    table = view.select("frame", indicator).read_all()
    data = table.to_pydict()
    frames = [
        int(frame)
        for frame, value in zip(data["frame"], data[indicator])
        if frame is not None and value is not None
    ]
    if not frames:
        raise ValueError(f"recording has no Points3D samples for {robot_id}")
    return frames[-1]


def _read_final_points(recording: rdf.Recording, robot_id: str):
    frame = _last_points_frame(recording, robot_id)
    entity = f"world/robots/{robot_id}/map/points"
    view = recording.view(index="frame", contents=f"{entity}/**")
    data = view.filter_range_sequence(frame, frame).select().read_all().to_pydict()
    positions = data[f"/{entity}:Position3D"]
    colors = data[f"/{entity}:Color"]
    if not positions or positions[0] is None:
        raise ValueError(f"missing final positions for {robot_id} at frame {frame}")
    if not colors or colors[0] is None:
        raise ValueError(f"missing final colors for {robot_id} at frame {frame}")
    points = np.asarray(positions[0], dtype=np.float64)
    packed_colors = np.asarray(colors[0], dtype=np.uint32)
    if points.shape != (len(packed_colors), 3):
        raise ValueError(
            f"point/color mismatch for {robot_id}: {points.shape} vs "
            f"{packed_colors.shape}"
        )
    return frame, points, packed_colors


def _read_final_session_from_map(
    recording: rdf.Recording,
    robot_id: str,
    frame: int,
):
    entity = f"world/robots/{robot_id}/map"
    data = (
        recording.view(index="frame", contents=entity)
        .filter_range_sequence(frame, frame)
        .select()
        .read_all()
        .to_pydict()
    )
    matrices = data[f"/{entity}:TransformMat3x3"]
    translations = data[f"/{entity}:Translation3D"]
    if not matrices or matrices[0] is None:
        raise ValueError(
            f"missing session_from_map matrix for {robot_id} at frame {frame}"
        )
    if not translations or translations[0] is None:
        raise ValueError(
            f"missing session_from_map translation for {robot_id} at frame {frame}"
        )
    matrix = np.asarray(matrices[0], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translations[0], dtype=np.float64).reshape(3)
    scale, orthogonality_error = _validate_session_matrix(matrix, robot_id)
    return matrix, translation, scale, orthogonality_error


def _validate_session_matrix(matrix, robot_id):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError(f"invalid session_from_map matrix for {robot_id}")
    scale = float(np.cbrt(determinant))
    rotation = matrix / scale
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    if orthogonality_error > 1e-4:
        raise ValueError(
            f"session_from_map is not a similarity for {robot_id}: "
            f"orthogonality error {orthogonality_error:.3e}"
        )
    return scale, orthogonality_error


def _read_npz_points(path: Path, robot_id: str, expected_keyframes: int):
    with np.load(path) as data:
        required = {
            "points",
            "colors",
            "session_from_map",
            "keyframe_count",
            "keyframe_input_indices",
            "poses",
            "patches_per_keyframe",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing NPZ fields: {sorted(missing)}")
        points = np.asarray(data["points"], dtype=np.float64)
        rgb = np.asarray(data["colors"], dtype=np.uint8)
        session_from_map = np.asarray(data["session_from_map"], dtype=np.float64)
        keyframe_count = int(data["keyframe_count"])
        keyframe_input_indices = np.asarray(
            data["keyframe_input_indices"], dtype=np.int64
        )
        input_poses = np.asarray(data["poses"], dtype=np.float64)
        declared_patches = int(data["patches_per_keyframe"])
    if keyframe_count <= 0:
        raise ValueError(f"{robot_id} NPZ has no keyframes")
    if len(keyframe_input_indices) != keyframe_count:
        raise ValueError(
            f"{robot_id} has {len(keyframe_input_indices)} keyframe indices for "
            f"{keyframe_count} keyframes"
        )
    if np.any(keyframe_input_indices < 0) or np.any(
        keyframe_input_indices >= len(input_poses)
    ):
        raise ValueError(f"{robot_id} NPZ has out-of-range keyframe indices")
    if points.shape != (len(rgb), 3) or rgb.shape[1:] != (3,):
        raise ValueError(
            f"invalid point/color arrays for {robot_id}: {points.shape}, {rgb.shape}"
        )
    if len(points) != keyframe_count * declared_patches:
        raise ValueError(
            f"{robot_id} NPZ declares {declared_patches} patches/keyframe but has "
            f"{len(points)} points for {keyframe_count} keyframes"
        )
    if session_from_map.shape != (4, 4):
        raise ValueError(
            f"invalid session_from_map shape for {robot_id}: "
            f"{session_from_map.shape}"
        )
    matrix = session_from_map[:3, :3]
    translation = session_from_map[:3, 3]
    scale, orthogonality_error = _validate_session_matrix(matrix, robot_id)
    session_rotation = matrix / scale
    source_pose_rows = input_poses[keyframe_input_indices]
    source_positions = source_pose_rows[:, :3] @ matrix.T + translation
    source_rotations = (
        session_rotation[None]
        @ Rotation.from_quat(source_pose_rows[:, 3:]).as_matrix()
    )
    source_poses = [
        Sim3(position, rotation, 1.0)
        for position, rotation in zip(source_positions, source_rotations)
    ]
    return (
        points,
        _pack_rgb(rgb),
        matrix,
        translation,
        scale,
        orthogonality_error,
        declared_patches,
        source_poses,
        expected_keyframes,
    )


def _unpack_rgb(packed_colors: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed_colors, dtype=np.uint32)
    return np.stack(
        [
            (packed >> 24) & 0xFF,
            (packed >> 16) & 0xFF,
            (packed >> 8) & 0xFF,
        ],
        axis=1,
    ).astype(np.uint8)


def _pack_rgb(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.uint32).reshape(-1, 3)
    return (
        (colors[:, 0] << 24)
        | (colors[:, 1] << 16)
        | (colors[:, 2] << 8)
        | np.uint32(255)
    ).astype(np.uint32)


def _tint_by_robot(packed_colors: np.ndarray, robot_id: str) -> np.ndarray:
    rgb = _unpack_rgb(packed_colors).astype(np.float64)
    luminance = np.clip(rgb.mean(axis=1, keepdims=True) / 255.0, 0.0, 1.0)
    base = ROBOT_COLORS.get(
        robot_id,
        np.array([220, 180, 70], dtype=np.uint8),
    ).astype(np.float64)
    tinted = np.clip(base[None, :] * (0.25 + 0.75 * luminance), 0, 255)
    return _pack_rgb(tinted.astype(np.uint8))


def _fit_sim3(source_positions, target_positions):
    """Fit target ~= scale * rotation * source + translation (Umeyama)."""

    source = np.asarray(source_positions, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_positions, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError(
            "Sim(3) fit needs at least three paired source/target positions"
        )
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        signs[-1] = -1.0
    rotation = u @ np.diag(signs) @ vt
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if not np.isfinite(source_variance) or source_variance <= 1e-12:
        raise ValueError("source trajectory is degenerate for Sim(3) fitting")
    scale = float(np.sum(singular_values * signs) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"fitted an invalid map scale: {scale}")
    translation = target_mean - scale * (rotation @ source_mean)
    transform = Sim3(translation, rotation, scale)
    errors = np.linalg.norm(transform.apply(source) - target, axis=1)
    return transform, errors


def _map_align_robot_points(
    robot_id,
    vertices,
    estimates,
    points,
    session_matrix,
    session_translation,
    source_poses=None,
    host_distance_quantile=0.995,
):
    """Place a complete local map with one fitted Sim(3), as online Rerun does."""

    vertices = sorted(vertices, key=lambda vertex: vertex.keyframe_id)
    keyframe_ids = [vertex.keyframe_id for vertex in vertices]
    if keyframe_ids != list(range(len(vertices))):
        raise ValueError(f"{robot_id} keyframe IDs are not dense from zero")
    if source_poses is None:
        source_poses = [vertex.estimate for vertex in vertices]
        target_indices = np.arange(len(vertices), dtype=np.int32)
        correspondence_mode = "exact_exported_keyframe_id"
    else:
        if not source_poses:
            raise ValueError(f"{robot_id} has no regenerated source poses")
        target_indices = np.rint(
            np.linspace(0, len(vertices) - 1, len(source_poses))
        ).astype(np.int32)
        correspondence_mode = "monotonic_normalized_sequence_progress"

    if len(points) % len(source_poses):
        raise ValueError(
            f"{robot_id} has {len(points)} points for {len(source_poses)} source "
            "keyframes"
        )
    patches_per_keyframe = len(points) // len(source_poses)
    if patches_per_keyframe <= 0:
        raise ValueError(f"{robot_id} has no points per keyframe")

    target_poses = []
    for target_index in target_indices:
        vertex = vertices[int(target_index)]
        key = (vertex.robot_id, vertex.session_id, vertex.keyframe_id)
        optimized = estimates.get(key)
        if optimized is None:
            raise ValueError(f"pose solution is missing {key}")
        target_poses.append(optimized)
    map_transform, fit_errors = _fit_sim3(
        [pose.translation for pose in source_poses],
        [pose.translation for pose in target_poses],
    )

    stable_points = points @ session_matrix.T + session_translation
    host_positions = np.repeat(
        np.asarray([pose.translation for pose in source_poses]),
        patches_per_keyframe,
        axis=0,
    )
    host_distances = np.linalg.norm(stable_points - host_positions, axis=1)
    finite_mask = np.isfinite(stable_points).all(axis=1) & np.isfinite(host_distances)
    finite_distances = host_distances[finite_mask]
    if not len(finite_distances):
        raise ValueError(f"{robot_id} has no finite source points")
    if not 0.0 < host_distance_quantile <= 1.0:
        raise ValueError("host-distance quantile must be in (0, 1]")
    host_distance_cutoff = float(
        np.quantile(finite_distances, host_distance_quantile)
    )
    retained_mask = finite_mask & (host_distances <= host_distance_cutoff)
    aligned = map_transform.apply(stable_points[retained_mask])
    keyframe_for_point = np.repeat(
        target_indices,
        patches_per_keyframe,
    )[retained_mask]
    target_positions = np.asarray(
        [pose.translation for pose in target_poses], dtype=np.float64
    )
    target_extent = float(
        np.linalg.norm(target_positions.max(axis=0) - target_positions.min(axis=0))
    )
    fit_rmse = float(np.sqrt(np.mean(fit_errors**2)))
    return aligned, keyframe_for_point, retained_mask, {
        "warp_mode": "map-level",
        "correspondence_mode": correspondence_mode,
        "source_keyframes": len(source_poses),
        "target_graph_keyframes": len(vertices),
        "matched_target_keyframes": int(len(np.unique(target_indices))),
        "patches_per_keyframe": patches_per_keyframe,
        "source_points": len(points),
        "retained_points": int(np.count_nonzero(retained_mask)),
        "host_distance_quantile": host_distance_quantile,
        "host_distance_cutoff": host_distance_cutoff,
        "map_transform": sim3_to_dict(map_transform),
        "trajectory_fit_rmse": fit_rmse,
        "trajectory_fit_median": float(np.median(fit_errors)),
        "trajectory_fit_p95": float(np.quantile(fit_errors, 0.95)),
        "trajectory_target_extent": target_extent,
        "trajectory_relative_rmse": (
            fit_rmse / target_extent if target_extent > 0.0 else None
        ),
    }


def _warp_robot_points(
    robot_id,
    vertices,
    estimates,
    points,
    session_matrix,
    session_translation,
    source_poses=None,
    host_distance_quantile=0.995,
):
    vertices = sorted(vertices, key=lambda vertex: vertex.keyframe_id)
    keyframe_ids = [vertex.keyframe_id for vertex in vertices]
    if keyframe_ids != list(range(len(vertices))):
        raise ValueError(f"{robot_id} keyframe IDs are not dense from zero")
    if source_poses is None:
        source_poses = [vertex.estimate for vertex in vertices]
        target_indices = np.arange(len(vertices), dtype=np.int32)
        correspondence_mode = "exact_exported_keyframe_id"
    else:
        if not source_poses:
            raise ValueError(f"{robot_id} has no regenerated source poses")
        target_indices = np.rint(
            np.linspace(0, len(vertices) - 1, len(source_poses))
        ).astype(np.int32)
        correspondence_mode = "monotonic_normalized_sequence_progress"
    if len(points) % len(source_poses):
        raise ValueError(
            f"{robot_id} has {len(points)} points for {len(source_poses)} source "
            "keyframes"
        )
    patches_per_keyframe = len(points) // len(source_poses)
    if patches_per_keyframe <= 0:
        raise ValueError(f"{robot_id} has no points per keyframe")

    stable_points = points @ session_matrix.T + session_translation
    warped = np.empty_like(stable_points)
    keyframe_for_point = np.empty(len(points), dtype=np.int32)
    host_positions = np.repeat(
        np.asarray([pose.translation for pose in source_poses]),
        patches_per_keyframe,
        axis=0,
    )
    host_distances = np.linalg.norm(stable_points - host_positions, axis=1)
    finite_mask = np.isfinite(stable_points).all(axis=1) & np.isfinite(host_distances)
    finite_distances = host_distances[finite_mask]
    if not len(finite_distances):
        raise ValueError(f"{robot_id} has no finite source points")
    if not 0.0 < host_distance_quantile <= 1.0:
        raise ValueError("host-distance quantile must be in (0, 1]")
    host_distance_cutoff = float(
        np.quantile(finite_distances, host_distance_quantile)
    )
    retained_mask = finite_mask & (host_distances <= host_distance_cutoff)
    max_translation_error = 0.0
    max_rotation_error = 0.0
    max_log_scale_error = 0.0

    for index, (source_pose, target_index) in enumerate(
        zip(source_poses, target_indices)
    ):
        vertex = vertices[int(target_index)]
        key = (vertex.robot_id, vertex.session_id, vertex.keyframe_id)
        optimized = estimates.get(key)
        if optimized is None:
            raise ValueError(f"CBS output is missing {key}")
        deformation = optimized.compose(source_pose.inverse())
        reproduced = deformation.compose(source_pose)
        delta = optimized.inverse().compose(reproduced)
        max_translation_error = max(
            max_translation_error,
            float(np.linalg.norm(delta.translation)),
        )
        max_rotation_error = max(
            max_rotation_error,
            float(Rotation.from_matrix(delta.rotation).magnitude()),
        )
        max_log_scale_error = max(
            max_log_scale_error,
            abs(float(np.log(delta.scale))),
        )

        begin = index * patches_per_keyframe
        end = begin + patches_per_keyframe
        warped[begin:end] = deformation.apply(stable_points[begin:end])
        keyframe_for_point[begin:end] = int(target_index)

    if not np.isfinite(warped[retained_mask]).all():
        raise ValueError(f"warped {robot_id} points contain non-finite values")
    return warped[retained_mask], keyframe_for_point[retained_mask], retained_mask, {
        "warp_mode": "posewise",
        "correspondence_mode": correspondence_mode,
        "source_keyframes": len(source_poses),
        "target_graph_keyframes": len(vertices),
        "matched_target_keyframes": int(len(np.unique(target_indices))),
        "patches_per_keyframe": patches_per_keyframe,
        "source_points": len(points),
        "retained_points": int(np.count_nonzero(retained_mask)),
        "host_distance_quantile": host_distance_quantile,
        "host_distance_cutoff": host_distance_cutoff,
        "max_pose_reproduction_translation_error": max_translation_error,
        "max_pose_reproduction_rotation_error_radians": max_rotation_error,
        "max_pose_reproduction_log_scale_error": max_log_scale_error,
    }


def _write_ply(path, points_by_robot, colors_by_robot, keyframes_by_robot):
    rows = []
    for robot_id in sorted(points_by_robot, key=_robot_sort_key):
        points = points_by_robot[robot_id].astype(np.float32)
        colors = _unpack_rgb(colors_by_robot[robot_id])
        keyframes = keyframes_by_robot[robot_id]
        robot_index = _robot_index(robot_id)
        robot_indices = np.full(len(points), robot_index, dtype=np.uint8)
        block = np.empty(
            len(points),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("robot_id", "u1"),
                ("keyframe_id", "<i4"),
            ],
        )
        block["x"], block["y"], block["z"] = points.T
        block["red"], block["green"], block["blue"] = colors.T
        block["robot_id"] = robot_indices
        block["keyframe_id"] = keyframes
        rows.append(block)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(np.concatenate(rows), "vertex")], text=False).write(
        path
    )


def _method_slug(label):
    slug = "".join(character.lower() if character.isalnum() else "_" for character in label)
    return slug.strip("_") or "solution"


def _log_rerun(
    output_path,
    recording_id,
    point_radius,
    graph,
    solutions,
    points_by_solution,
    colors_by_robot,
):
    labels = list(solutions)
    if len(labels) == 1:
        label = labels[0]
        slug = _method_slug(label)
        layout = rrb.Horizontal(
            rrb.Spatial3DView(
                name=f"{label} joint sparse map",
                origin="world",
                contents=[
                    f"world/solutions/{slug}/rgb/**",
                    f"world/solutions/{slug}/pose_graph/**",
                ],
            ),
            rrb.Spatial3DView(
                name=f"{label} map ownership",
                origin="world",
                contents=[
                    f"world/solutions/{slug}/by_robot/**",
                    f"world/solutions/{slug}/pose_graph/**",
                ],
            ),
        )
    else:
        layout = rrb.Horizontal(
            *[
                rrb.Spatial3DView(
                    name=label,
                    origin="world",
                    contents=[
                        f"world/solutions/{_method_slug(label)}/by_robot/**",
                        f"world/solutions/{_method_slug(label)}/pose_graph/**",
                    ],
                )
                for label in labels
            ]
        )
    blueprint = rrb.Blueprint(layout, collapse_panels=True)
    rr.init(
        "DPVO CBS Joint Map",
        recording_id=recording_id,
        default_blueprint=blueprint,
        strict=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(output_path)
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    vertex_by_id = {vertex.vertex_id: vertex for vertex in graph.vertices}
    for label, estimates in solutions.items():
        slug = _method_slug(label)
        for robot_id in sorted(points_by_solution[label], key=_robot_sort_key):
            points = points_by_solution[label][robot_id].astype(np.float32)
            colors = colors_by_robot[robot_id]
            rr.log(
                f"world/solutions/{slug}/rgb/{robot_id}",
                rr.Points3D(
                    points,
                    colors=colors,
                    radii=rr.Radius.ui_points(point_radius),
                ),
                static=True,
            )
            rr.log(
                f"world/solutions/{slug}/by_robot/{robot_id}",
                rr.Points3D(
                    points,
                    colors=_tint_by_robot(colors, robot_id),
                    radii=rr.Radius.ui_points(point_radius),
                ),
                static=True,
            )

            robot_vertices = sorted(
                (vertex for vertex in graph.vertices if vertex.robot_id == robot_id),
                key=lambda vertex: vertex.keyframe_id,
            )
            positions = np.asarray(
                [
                    estimates[(v.robot_id, v.session_id, v.keyframe_id)].translation
                    for v in robot_vertices
                ],
                dtype=np.float32,
            )
            color = ROBOT_COLORS.get(robot_id, np.array([220, 180, 70]))
            rr.log(
                f"world/solutions/{slug}/pose_graph/trajectories/{robot_id}",
                rr.LineStrips3D(
                    [positions],
                    colors=color,
                    radii=rr.Radius.ui_points(2.5),
                ),
                static=True,
            )
            rr.log(
                f"world/solutions/{slug}/pose_graph/keyframes/{robot_id}",
                rr.Points3D(
                    positions,
                    colors=color,
                    radii=rr.Radius.ui_points(2.0),
                ),
                static=True,
            )

        loop_strips = []
        loop_labels = []
        for edge in graph.edges:
            if edge.edge_type != "inter_robot_loop_closure":
                continue
            source = vertex_by_id[edge.source]
            target = vertex_by_id[edge.target]
            source_pose = estimates[
                (source.robot_id, source.session_id, source.keyframe_id)
            ]
            target_pose = estimates[
                (target.robot_id, target.session_id, target.keyframe_id)
            ]
            loop_strips.append(
                np.asarray(
                    [source_pose.translation, target_pose.translation],
                    dtype=np.float32,
                )
            )
            loop_labels.append(
                f"{source.robot_id}:{source.keyframe_id} -> "
                f"{target.robot_id}:{target.keyframe_id}"
            )
        if loop_strips:
            rr.log(
                f"world/solutions/{slug}/pose_graph/inter_robot_loops",
                rr.LineStrips3D(
                    loop_strips,
                    colors=[[255, 190, 50]] * len(loop_strips),
                    labels=loop_labels,
                    radii=rr.Radius.ui_points(3.0),
                ),
                static=True,
            )
    rr.disconnect()


def main():
    args = _parse_args()
    if not 0.0 < args.host_distance_quantile <= 1.0:
        raise ValueError("--host-distance-quantile must be in (0, 1]")
    primary_label = args.primary_label.strip()
    if not primary_label:
        raise ValueError("--primary-label must not be empty")
    if primary_label == "Online centralized":
        raise ValueError("solution label 'Online centralized' is reserved")
    npz_sources = _parse_npz_sources(args.input_npz)
    comparison_sources = _parse_labeled_paths(
        args.comparison_csv, {primary_label}
    )
    input_paths = [args.input_graph, args.cbs_csv, *comparison_sources.values()]
    if args.online_map_transforms_json is not None:
        input_paths.append(args.online_map_transforms_json)
    if args.input_rrd is not None:
        input_paths.append(args.input_rrd)
    else:
        input_paths.extend(npz_sources.values())
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    graph = read_json(args.input_graph)
    solutions = {primary_label: _read_cbs_csv(args.cbs_csv)}
    solutions.update(
        {
            label: _read_cbs_csv(path)
            for label, path in comparison_sources.items()
        }
    )
    if args.online_map_transforms_json is not None:
        if "Online centralized" in solutions:
            raise ValueError("solution label 'Online centralized' is reserved")
        solutions["Online centralized"] = _read_online_map_estimates(
            args.online_map_transforms_json,
            graph,
        )
    graph_keys = {
        (vertex.robot_id, vertex.session_id, vertex.keyframe_id)
        for vertex in graph.vertices
    }
    for label, estimates in solutions.items():
        missing = graph_keys - estimates.keys()
        if missing:
            raise ValueError(
                f"{label} output is missing {len(missing)} graph vertices"
            )

    robots = sorted(
        {vertex.robot_id for vertex in graph.vertices}, key=_robot_sort_key
    )
    if args.input_rrd is None and set(npz_sources) != set(robots):
        raise ValueError(
            "--input-npz robot set does not match graph: "
            f"got {sorted(npz_sources)}, expected {robots}"
        )
    recording = (
        _load_input_recording(args.input_rrd, args.input_recording_id)
        if args.input_rrd
        else None
    )
    points_by_solution = {label: {} for label in solutions}
    colors_by_robot = {}
    keyframes_by_robot = {}
    manifest_robots = {}
    warp_function = (
        _map_align_robot_points
        if args.warp_mode == "map-level"
        else _warp_robot_points
    )

    for robot_id in robots:
        vertices = [v for v in graph.vertices if v.robot_id == robot_id]
        if recording is not None:
            frame, points, colors = _read_final_points(recording, robot_id)
            matrix, translation, session_scale, orthogonality_error = (
                _read_final_session_from_map(recording, robot_id, frame)
            )
            source_patches = None
            source_poses = None
        else:
            frame = None
            (
                points,
                colors,
                matrix,
                translation,
                session_scale,
                orthogonality_error,
                source_patches,
                source_poses,
                _expected_keyframes,
            ) = _read_npz_points(npz_sources[robot_id], robot_id, len(vertices))
        solution_validation = {}
        reference_mask = None
        for label, estimates in solutions.items():
            warped, keyframe_ids, retained_mask, validation = warp_function(
                robot_id,
                vertices,
                estimates,
                points,
                matrix,
                translation,
                source_poses=source_poses,
                host_distance_quantile=args.host_distance_quantile,
            )
            points_by_solution[label][robot_id] = warped
            solution_validation[label] = validation
            if reference_mask is None:
                reference_mask = retained_mask
                colors_by_robot[robot_id] = colors[retained_mask]
                keyframes_by_robot[robot_id] = keyframe_ids
            elif not np.array_equal(reference_mask, retained_mask):
                raise RuntimeError(
                    f"point filter changed between solutions for {robot_id}"
                )
            print(
                f"{label} {robot_id}: {len(vertices)} keyframes, "
                f"{len(warped)} points, "
                f"{validation['patches_per_keyframe']} patches/keyframe"
            )
        manifest_robots[robot_id] = {
            "keyframes": len(vertices),
            "points": int(np.count_nonzero(reference_mask)),
            "source_rrd_final_frame": frame,
            "session_from_map_scale": session_scale,
            "session_from_map_orthogonality_error": orthogonality_error,
            "solutions": solution_validation,
        }
        if source_patches is not None:
            manifest_robots[robot_id]["source_npz_patches_per_keyframe"] = (
                source_patches
            )

    _log_rerun(
        args.output_rrd,
        args.recording_id,
        args.point_radius,
        graph,
        solutions,
        points_by_solution,
        colors_by_robot,
    )
    if args.output_ply is not None:
        _write_ply(
            args.output_ply,
            points_by_solution[primary_label],
            colors_by_robot,
            keyframes_by_robot,
        )

    manifest_path = args.output_manifest or args.output_rrd.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "dpvo_cbs_joint_sparse_map",
        "version": 2,
        "coordinate_frame": "solution frames from the supplied pose estimates",
        "warp_mode": args.warp_mode,
        "warp": (
            "one least-squares Sim(3) from source DPVO map to each solution"
            if args.warp_mode == "map-level"
            else (
                "Y_solution[k] * inverse(X_source[k]) * "
                "T_session_map * p_internal"
            )
        ),
        "host_distance_quantile": args.host_distance_quantile,
        "inputs": {
            "rrd": str(args.input_rrd.resolve()) if args.input_rrd else None,
            "rrd_recording_id": (
                str(recording.recording_id()) if recording is not None else None
            ),
            "npz": {
                robot_id: str(path.resolve())
                for robot_id, path in sorted(
                    npz_sources.items(), key=lambda item: _robot_sort_key(item[0])
                )
            },
            "graph": str(args.input_graph.resolve()),
            "cbs_csv": str(args.cbs_csv.resolve()),
            "primary_label": primary_label,
            "comparison_csv": {
                label: str(path.resolve())
                for label, path in comparison_sources.items()
            },
            "online_map_transforms_json": (
                str(args.online_map_transforms_json.resolve())
                if args.online_map_transforms_json
                else None
            ),
        },
        "outputs": {
            "rrd": str(args.output_rrd.resolve()),
            "ply": str(args.output_ply.resolve()) if args.output_ply else None,
        },
        "total_keyframes": len(graph.vertices),
        "total_points_per_solution": {
            label: int(sum(len(points) for points in points_by_robot.values()))
            for label, points_by_robot in points_by_solution.items()
        },
        "inter_robot_loops": sum(
            edge.edge_type == "inter_robot_loop_closure" for edge in graph.edges
        ),
        "solutions": list(solutions),
        "robots": manifest_robots,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote Rerun recording: {args.output_rrd}")
    if args.output_ply is not None:
        print(f"Wrote joint point cloud: {args.output_ply}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

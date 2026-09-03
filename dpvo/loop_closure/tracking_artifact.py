"""Persistent DPVO tracking artifacts for staged multi-robot experiments.

Stage one records only per-robot odometry state and the images needed by the
later geometric-verification stage.  In particular, this format contains no
inter-robot constraints and no optimized global or anchor poses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .centralized import CentralizedPgoResult, Sim3
from .pose_graph import (
    build_keyframe_graph,
    split_keyframe_graph_by_robot,
    write_g2o,
    write_json,
)


FORMAT_NAME = "dpvo_tracking_artifact"
FORMAT_VERSION = 2


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_set_sha256(root: Path, input_indices) -> str:
    """Hash each referenced keyframe image and its input-frame identity."""

    digest = hashlib.sha256()
    for input_index in np.asarray(input_indices, dtype=np.int64):
        frame_name = f"{int(input_index):08d}.jpg"
        frame_path = root / "frames" / frame_name
        digest.update(frame_name.encode("ascii"))
        with frame_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrackingArtifact:
    root: Path
    robot_id: str
    session_id: str
    keyframe_input_indices: np.ndarray
    keyframe_timestamps: np.ndarray
    keyframe_poses_xyzw: np.ndarray
    internal_poses_xyzw: np.ndarray
    internal_intrinsics: np.ndarray
    patch_disparities: np.ndarray
    session_from_map: np.ndarray
    input_poses_xyzw: np.ndarray
    map_points: np.ndarray
    map_colors: np.ndarray
    patches_per_keyframe: int
    image_width: int
    image_height: int
    dpvo_resolution: int
    manifest: dict

    @property
    def keyframe_count(self) -> int:
        return int(self.keyframe_input_indices.size)

    def frame_path(self, keyframe_id: int) -> Path:
        if keyframe_id < 0 or keyframe_id >= self.keyframe_count:
            raise IndexError(f"invalid keyframe {keyframe_id} for {self.robot_id}")
        input_index = int(self.keyframe_input_indices[keyframe_id])
        return self.root / "frames" / f"{input_index:08d}.jpg"

    def validate(self) -> "TrackingArtifact":
        count = self.keyframe_count
        expected = {
            "keyframe_timestamps": (count,),
            "keyframe_poses_xyzw": (count, 7),
            "internal_poses_xyzw": (count, 7),
            "internal_intrinsics": (count, 4),
            "patch_disparities": (count,),
            "session_from_map": (4, 4),
            "map_points": (count * self.patches_per_keyframe, 3),
            "map_colors": (count * self.patches_per_keyframe, 3),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(
                    f"{self.robot_id} artifact {name} has shape "
                    f"{np.asarray(getattr(self, name)).shape}, expected {shape}"
                )
        if count < 1:
            raise ValueError(f"{self.robot_id} artifact has no keyframes")
        if self.patches_per_keyframe < 1:
            raise ValueError(
                f"{self.robot_id} artifact has invalid patches-per-keyframe "
                f"{self.patches_per_keyframe}"
            )
        if (
            self.input_poses_xyzw.ndim != 2
            or self.input_poses_xyzw.shape[1:] != (7,)
        ):
            raise ValueError(
                f"{self.robot_id} artifact input_poses_xyzw has shape "
                f"{self.input_poses_xyzw.shape}, expected (N, 7)"
            )
        if np.any(self.keyframe_input_indices < 0) or np.any(
            self.keyframe_input_indices >= len(self.input_poses_xyzw)
        ):
            raise ValueError(f"{self.robot_id} has out-of-range keyframe indices")
        if np.any(self.patch_disparities <= 0) or not np.isfinite(
            self.patch_disparities
        ).all():
            raise ValueError(f"{self.robot_id} has invalid patch disparities")
        if not np.isfinite(self.keyframe_poses_xyzw).all():
            raise ValueError(f"{self.robot_id} has non-finite poses")
        if not np.isfinite(self.input_poses_xyzw).all():
            raise ValueError(f"{self.robot_id} has non-finite input poses")
        if not np.isfinite(self.map_points).all(axis=1).any():
            raise ValueError(f"{self.robot_id} has no finite map points")
        missing = [
            str(self.frame_path(keyframe_id))
            for keyframe_id in range(count)
            if not self.frame_path(keyframe_id).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{self.robot_id} artifact is missing {len(missing)} frame(s); "
                f"first: {missing[0]}"
            )
        return self


def save_tracking_artifact(
    output_dir: Path,
    *,
    robot_id: str,
    session_id: str,
    keyframe_input_indices,
    keyframe_timestamps,
    keyframe_poses_xyzw,
    internal_poses_xyzw,
    internal_intrinsics,
    patch_disparities,
    session_from_map,
    input_poses_xyzw,
    map_points,
    map_colors,
    patches_per_keyframe: int,
    image_width: int,
    image_height: int,
    dpvo_resolution: int,
    input_frame_count: int,
    random_seed: int,
    config_path: str,
    network_path: str,
) -> Path:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.npz"
    temporary = output_dir / ".state.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            keyframe_input_indices=np.asarray(keyframe_input_indices, dtype=np.int64),
            keyframe_timestamps=np.asarray(keyframe_timestamps, dtype=np.float64),
            keyframe_poses_xyzw=np.asarray(keyframe_poses_xyzw, dtype=np.float32),
            internal_poses_xyzw=np.asarray(internal_poses_xyzw, dtype=np.float32),
            internal_intrinsics=np.asarray(internal_intrinsics, dtype=np.float32),
            patch_disparities=np.asarray(patch_disparities, dtype=np.float32),
            session_from_map=np.asarray(session_from_map, dtype=np.float64),
        )
    temporary.replace(state_path)
    map_path = output_dir / "map.npz"
    map_temporary = output_dir / ".map.npz.tmp"
    with map_temporary.open("wb") as stream:
        # Keep this schema compatible with visualize_cbs_joint_map.py and the
        # standalone DPVO NPZ exporter. Points and poses remain in DPVO's
        # mutable internal chart; session_from_map moves them into the stable
        # original-scale robot-local chart.
        np.savez_compressed(
            stream,
            poses=np.asarray(input_poses_xyzw, dtype=np.float32),
            points=np.asarray(map_points, dtype=np.float32),
            colors=np.asarray(map_colors, dtype=np.uint8),
            session_from_map=np.asarray(session_from_map, dtype=np.float64),
            keyframe_count=np.asarray(
                np.asarray(keyframe_input_indices).size, dtype=np.int64
            ),
            keyframe_input_indices=np.asarray(
                keyframe_input_indices, dtype=np.int64
            ),
            patches_per_keyframe=np.asarray(
                patches_per_keyframe, dtype=np.int64
            ),
        )
    map_temporary.replace(map_path)
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "complete": True,
        "robot_id": robot_id,
        "session_id": session_id,
        "coordinate_frame": "stable_robot_local_map_original_scale",
        "contains_inter_robot_constraints": False,
        "contains_global_optimization": False,
        "contains_full_sparse_map": True,
        "keyframe_count": int(np.asarray(keyframe_input_indices).size),
        "input_frame_count": int(input_frame_count),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "dpvo_resolution": int(dpvo_resolution),
        "random_seed": int(random_seed),
        "config_path": str(config_path),
        "network_path": str(network_path),
        "frame_pattern": "frames/{input_index:08d}.jpg",
        "state_file": state_path.name,
        "state_sha256": _sha256(state_path),
        "map_file": map_path.name,
        "map_sha256": _sha256(map_path),
        "keyframe_images_sha256": _frame_set_sha256(
            output_dir, keyframe_input_indices
        ),
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return output_dir / "manifest.json"


def write_incomplete_manifest(
    output_dir: Path, *, robot_id: str, session_id: str
) -> None:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "complete": False,
        "robot_id": robot_id,
        "session_id": session_id,
        "contains_inter_robot_constraints": False,
        "contains_global_optimization": False,
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def load_tracking_artifact(path: Path) -> TrackingArtifact:
    path = Path(path).expanduser()
    root = path.parent if path.is_file() else path
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"not a DPVO tracking artifact: {manifest_path}")
    if int(manifest.get("version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported tracking artifact version: {manifest}")
    if not manifest.get("complete", False):
        raise RuntimeError(f"tracking artifact is incomplete: {root}")
    if manifest.get("contains_inter_robot_constraints", True):
        raise ValueError("stage-one artifact unexpectedly contains loop constraints")
    if manifest.get("contains_global_optimization", True):
        raise ValueError("stage-one artifact unexpectedly contains global optimization")
    state_path = root / manifest["state_file"]
    if _sha256(state_path) != manifest["state_sha256"]:
        raise ValueError(f"tracking state checksum mismatch: {state_path}")
    map_path = root / manifest["map_file"]
    if _sha256(map_path) != manifest["map_sha256"]:
        raise ValueError(f"tracking map checksum mismatch: {map_path}")
    with np.load(state_path, allow_pickle=False) as state, np.load(
        map_path, allow_pickle=False
    ) as map_state:
        artifact = TrackingArtifact(
            root=root,
            robot_id=str(manifest["robot_id"]),
            session_id=str(manifest["session_id"]),
            keyframe_input_indices=state["keyframe_input_indices"].copy(),
            keyframe_timestamps=state["keyframe_timestamps"].copy(),
            keyframe_poses_xyzw=state["keyframe_poses_xyzw"].copy(),
            internal_poses_xyzw=state["internal_poses_xyzw"].copy(),
            internal_intrinsics=state["internal_intrinsics"].copy(),
            patch_disparities=state["patch_disparities"].copy(),
            session_from_map=state["session_from_map"].copy(),
            input_poses_xyzw=map_state["poses"].copy(),
            map_points=map_state["points"].copy(),
            map_colors=map_state["colors"].copy(),
            patches_per_keyframe=int(map_state["patches_per_keyframe"]),
            image_width=int(manifest["image_width"]),
            image_height=int(manifest["image_height"]),
            dpvo_resolution=int(manifest["dpvo_resolution"]),
            manifest=manifest,
        )
    artifact.validate()
    if int(manifest.get("keyframe_count", -1)) != artifact.keyframe_count:
        raise ValueError(f"tracking artifact keyframe count mismatch: {root}")
    if manifest.get("keyframe_images_sha256") != _frame_set_sha256(
        root, artifact.keyframe_input_indices
    ):
        raise ValueError(f"tracking artifact image checksum mismatch: {root}")
    return artifact


def load_artifact_root(
    root: Path, robot_ids: list[str] | tuple[str, ...] | None = None
) -> dict[str, TrackingArtifact]:
    root = Path(root).expanduser()
    selected = None
    if robot_ids is not None:
        requested = [str(robot_id) for robot_id in robot_ids]
        if len(set(requested)) != len(requested):
            duplicates = sorted(
                robot_id
                for robot_id in set(requested)
                if requested.count(robot_id) > 1
            )
            raise ValueError("duplicate robot ids requested: " + ", ".join(duplicates))
        if len(requested) < 2:
            raise ValueError("at least two robot ids must be requested")
        selected = set(requested)

    artifacts = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if selected is not None:
            try:
                manifest_robot_id = str(
                    json.loads(manifest_path.read_text())["robot_id"]
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"cannot identify tracking artifact {manifest_path}: {error}"
                ) from error
            if manifest_robot_id not in selected:
                continue
        artifact = load_tracking_artifact(manifest_path)
        if artifact.robot_id in artifacts:
            raise ValueError(f"duplicate tracking artifact for {artifact.robot_id}")
        artifacts[artifact.robot_id] = artifact
    if selected is not None:
        missing = selected - set(artifacts)
        if missing:
            raise ValueError(
                "unknown or missing robot ids under "
                f"{root}: {', '.join(sorted(missing))}"
            )
    if len(artifacts) < 2:
        raise RuntimeError(f"expected at least two robot artifacts under {root}")
    return artifacts


def write_raw_pose_graph(
    artifacts: dict[str, TrackingArtifact],
    output_base: Path,
    *,
    anchor_robot_id: str = "robot0",
    odometry_weight: float = 100.0,
):
    paths = {
        robot_id: [Sim3.from_pose(pose) for pose in artifact.keyframe_poses_xyzw]
        for robot_id, artifact in artifacts.items()
    }
    graph = build_keyframe_graph(
        [],
        CentralizedPgoResult({}, True, 0.0, 0.0),
        paths,
        {robot_id: artifact.session_id for robot_id, artifact in artifacts.items()},
        anchor_robot_id,
        odometry_weight=odometry_weight,
        align_to_global=False,
        timestamps={
            robot_id: artifact.keyframe_timestamps.tolist()
            for robot_id, artifact in artifacts.items()
        },
    )
    graph.metadata.update(
        {
            "pipeline_stage": "tracking",
            "contains_inter_robot_constraints": False,
            "contains_global_optimization": False,
        }
    )
    output_base = Path(output_base).expanduser()
    write_json(graph, output_base.with_suffix(".json"))
    write_g2o(graph, output_base.with_suffix(".g2o"))
    for robot_id, robot_graph in split_keyframe_graph_by_robot(graph).items():
        robot_base = output_base.with_name(f"{output_base.name}_{robot_id}")
        write_json(robot_graph, robot_base.with_suffix(".json"))
        write_g2o(robot_graph, robot_base.with_suffix(".g2o"))
    return graph

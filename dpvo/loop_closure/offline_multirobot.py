"""Offline retrieval and geometric verification for staged DPVO runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import cv2
from einops import rearrange, repeat
import numpy as np
import torch
from torch_scatter import scatter_max

from .. import fastba
from .. import projective_ops as pops
from ..lietorch import SE3
from ..map_gauge import scale_camera_points
from .centralized import CentralizedPgoResult, RobotMapConstraint, Sim3
from .distributed import (
    BowCandidateDetector,
    FrameIdentity,
    KeyframePayload,
    RobustSim3Verifier,
    SparseBow,
    depth_statistics,
    valid_depth_mask,
)
from .pose_graph import (
    build_keyframe_graph,
    split_keyframe_graph_by_robot,
    write_g2o,
    write_json,
)
from .tracking_artifact import (
    TrackingArtifact,
    load_artifact_root,
    write_raw_pose_graph,
)


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _sparse_bow(entries) -> SparseBow:
    if not entries:
        return SparseBow([], [])
    word_ids, values = zip(*entries, strict=True)
    return SparseBow(word_ids, values)


def _write_bow_cache(
    path: Path,
    artifact: TrackingArtifact,
    bows: dict[int, SparseBow],
) -> None:
    keyframe_ids = np.asarray(sorted(bows), dtype=np.int64)
    offsets = [0]
    word_ids = []
    values = []
    for keyframe_id in keyframe_ids:
        bow = bows[int(keyframe_id)]
        word_ids.extend(bow.word_ids.tolist())
        values.extend(bow.word_values.tolist())
        offsets.append(len(word_ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            state_sha256=np.asarray(artifact.manifest["state_sha256"]),
            keyframe_ids=keyframe_ids,
            offsets=np.asarray(offsets, dtype=np.int64),
            word_ids=np.asarray(word_ids, dtype=np.uint32),
            values=np.asarray(values, dtype=np.float32),
        )
    temporary.replace(path)


def _read_bow_cache(
    path: Path, artifact: TrackingArtifact
) -> dict[int, SparseBow] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["state_sha256"].item()) != artifact.manifest["state_sha256"]:
            return None
        keyframe_ids = cache["keyframe_ids"]
        offsets = cache["offsets"]
        word_ids = cache["word_ids"]
        values = cache["values"]
        return {
            int(keyframe_id): SparseBow(
                word_ids[offsets[index] : offsets[index + 1]],
                values[offsets[index] : offsets[index + 1]],
            )
            for index, keyframe_id in enumerate(keyframe_ids)
        }


def compute_bow_vectors(
    artifact: TrackingArtifact,
    vocabulary_path: Path,
    cache_path: Path,
) -> dict[int, SparseBow]:
    cached = _read_bow_cache(cache_path, artifact)
    if cached is not None:
        return cached
    import dpretrieval

    database = dpretrieval.DPRetrieval(str(vocabulary_path), 50)
    bows = {}
    print(
        f"[BoW] {artifact.robot_id}: computing "
        f"{max(artifact.keyframe_count - 2, 0)} vectors",
        flush=True,
    )
    # Endpoints cannot provide the image triplet needed by geometric
    # verification, so they are deliberately absent from place retrieval.
    for keyframe_id in range(1, artifact.keyframe_count - 1):
        image = cv2.imread(str(artifact.frame_path(keyframe_id)), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {artifact.frame_path(keyframe_id)}")
        entries = database.insert_image(image)
        if entries is None:
            raise RuntimeError("dpretrieval binding does not export BoW vectors")
        bows[keyframe_id] = _sparse_bow(entries)
        if keyframe_id % 100 == 0:
            print(
                f"[BoW] {artifact.robot_id}: {keyframe_id}/"
                f"{artifact.keyframe_count - 2}",
                flush=True,
            )
    _write_bow_cache(cache_path, artifact, bows)
    return bows


class OfflineKeyframeExtractor:
    """Reconstruct the online DPVO 3-D local-feature payload on demand."""

    def __init__(self, backend: str = "disk", random_seed: int = 1234):
        self.backend = backend.lower()
        self.random_seed = int(random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        if self.backend == "disk":
            import kornia.feature as KF

            self.local_frontend = None
            self.detector = KF.DISK.from_pretrained("depth").cuda().eval()
            self.matcher = KF.LightGlue("disk").cuda().eval()
        elif self.backend == "xfeat":
            from .learned_frontend import XFeatFrontend

            self.local_frontend = XFeatFrontend(
                repo_or_dir="verlab/accelerated_features",
                top_k=2048,
                detection_threshold=0.05,
                min_confidence=0.10,
            )
            self.detector = self.local_frontend.model
            self.matcher = self.local_frontend.matcher
        else:
            raise ValueError("local feature backend must be 'disk' or 'xfeat'")

    def _detect(self, images):
        if self.local_frontend is not None:
            return self.local_frontend.detect(images)
        _, _, height, width = images.shape
        image_size = torch.tensor(
            [width, height], device="cuda", dtype=torch.float32
        ).view(1, 2)
        features = self.detector(
            images,
            2048,
            pad_if_not_divisible=True,
            window_size=15,
            score_threshold=40.0,
        )
        return [
            {
                "keypoints": feature.keypoints[None],
                "descriptors": feature.descriptors[None],
                "image_size": image_size,
            }
            for feature in features
        ]

    @staticmethod
    def _cache_path(cache_root: Path, artifact: TrackingArtifact, keyframe_id: int):
        return cache_root / artifact.robot_id / f"{keyframe_id:08d}.npz"

    @staticmethod
    def _load_payload_cache(
        path: Path, artifact: TrackingArtifact, keyframe_id: int
    ) -> KeyframePayload | None:
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["state_sha256"].item()) != artifact.manifest["state_sha256"]:
                return None
            return KeyframePayload(
                frame=FrameIdentity(
                    artifact.robot_id, artifact.session_id, keyframe_id
                ),
                timestamp=float(artifact.keyframe_timestamps[keyframe_id]),
                pose=cache["pose"].copy(),
                points=cache["points"].copy(),
                keypoints=cache["keypoints"].copy(),
                descriptors=cache["descriptors"].copy(),
                image_size=cache["image_size"].copy(),
            )

    @staticmethod
    def _save_payload_cache(
        path: Path, artifact: TrackingArtifact, payload: KeyframePayload
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                state_sha256=np.asarray(artifact.manifest["state_sha256"]),
                pose=payload.pose,
                points=payload.points,
                keypoints=payload.keypoints,
                descriptors=payload.descriptors,
                image_size=payload.image_size,
            )
        temporary.replace(path)

    @torch.inference_mode()
    def payload(
        self,
        artifact: TrackingArtifact,
        keyframe_id: int,
        cache_root: Path,
    ) -> KeyframePayload:
        if keyframe_id < 1 or keyframe_id >= artifact.keyframe_count - 1:
            raise IndexError("offline keyframe payload requires an image triplet")
        cache_path = self._cache_path(cache_root, artifact, keyframe_id)
        cached = self._load_payload_cache(cache_path, artifact, keyframe_id)
        if cached is not None:
            return cached

        image_list = [
            cv2.imread(str(artifact.frame_path(index)), cv2.IMREAD_COLOR)
            for index in (keyframe_id - 1, keyframe_id, keyframe_id + 1)
        ]
        if any(image is None for image in image_list):
            raise RuntimeError(
                f"failed to load image triplet for {artifact.robot_id}/{keyframe_id}"
            )
        images = (
            torch.from_numpy(np.stack(image_list))
            .permute(0, 3, 1, 2)
            .contiguous()
            .cuda()
            .float()
            / 255.0
        )
        features = self._detect(images)
        center_count = int(features[1]["keypoints"].shape[1])
        if center_count == 0:
            raise RuntimeError("local feature detector returned no features")
        trajectories = torch.full(
            (center_count, 3), -1, dtype=torch.long, device="cuda"
        )
        trajectories[:, 1] = torch.arange(center_count, device="cuda")
        first_match = self.matcher(
            {"image0": features[0], "image1": features[1]}
        )["matches"][0]
        if first_match.numel():
            first, center = first_match.mT
            trajectories[center, 0] = first
        second_match = self.matcher(
            {"image0": features[2], "image1": features[1]}
        )["matches"][0]
        if second_match.numel():
            second, center = second_match.mT
            trajectories[center, 2] = second
        trajectories = trajectories[
            torch.randperm(center_count, device="cuda")
        ]
        trajectories = trajectories[trajectories.min(dim=1).values >= 0]
        if trajectories.shape[0] < 3:
            raise RuntimeError("fewer than three triplet feature tracks")

        first, center, second = trajectories.mT
        keypoints0 = features[0]["keypoints"][:, first]
        keypoints1 = features[1]["keypoints"][:, center]
        keypoints2 = features[2]["keypoints"][:, second]
        descriptors1 = features[1]["descriptors"][:, center]
        count = int(trajectories.shape[0])
        kk = torch.arange(count, device="cuda").repeat(2)
        ii = torch.ones(2 * count, dtype=torch.long, device="cuda")
        jj = torch.zeros(2 * count, dtype=torch.long, device="cuda")
        jj[count:] = 2
        disparity = float(artifact.patch_disparities[keyframe_id])
        patches = torch.cat(
            (
                keypoints1,
                torch.full(
                    (1, count, 1),
                    disparity,
                    device="cuda",
                    dtype=keypoints1.dtype,
                ),
            ),
            dim=-1,
        )
        patches = repeat(patches, "1 n uvd -> 1 n uvd 3 3", uvd=3)
        target = rearrange(
            torch.stack((keypoints0, keypoints2)),
            "other 1 n uv -> 1 (other n) uv",
            other=2,
            n=count,
            uv=2,
        )
        weight = torch.ones_like(target)
        poses = torch.from_numpy(
            artifact.internal_poses_xyzw[keyframe_id - 1 : keyframe_id + 2]
        )[None].cuda()
        intrinsics = (
            torch.from_numpy(
                artifact.internal_intrinsics[keyframe_id - 1 : keyframe_id + 2]
            )[None].cuda()
            * artifact.dpvo_resolution
        )
        damping = torch.as_tensor([1e-3], device="cuda")
        fastba.BA(
            poses,
            patches,
            intrinsics,
            target,
            weight,
            damping,
            ii,
            jj,
            kk,
            3,
            3,
            M=-1,
            iterations=6,
            eff_impl=False,
        )
        coordinates = pops.transform(
            SE3(poses), patches, intrinsics, ii, jj, kk
        )[:, :, 1, 1]
        residual = (coordinates - target).norm(dim=-1).squeeze(0)
        valid = scatter_max(residual, kk)[0] < 2
        points = pops.iproj(
            patches,
            intrinsics[
                :, torch.ones(count, dtype=torch.long, device="cuda")
            ],
        )
        points = points[..., 1, 1, :3] / points[..., 1, 1, 3:]
        points = points[:, valid].squeeze(0).detach().float().cpu().numpy()
        points = scale_camera_points(artifact.session_from_map, points).astype(
            np.float32
        )
        payload = KeyframePayload(
            frame=FrameIdentity(
                artifact.robot_id, artifact.session_id, keyframe_id
            ),
            timestamp=float(artifact.keyframe_timestamps[keyframe_id]),
            pose=artifact.keyframe_poses_xyzw[keyframe_id].astype(np.float32),
            points=points,
            keypoints=(
                keypoints1[:, valid].squeeze(0).detach().float().cpu().numpy()
            ),
            descriptors=(
                descriptors1[:, valid]
                .squeeze(0)
                .detach()
                .float()
                .cpu()
                .numpy()
            ),
            image_size=(
                features[1]["image_size"]
                .squeeze(0)
                .detach()
                .float()
                .cpu()
                .numpy()
            ),
        )
        self._save_payload_cache(cache_path, artifact, payload)
        del images, features, poses, patches, target, coordinates
        torch.cuda.empty_cache()
        return payload

    @torch.inference_mode()
    def match(self, first: KeyframePayload, second: KeyframePayload, max_depth: float):
        first_mask = valid_depth_mask(first.points, max_depth)
        second_mask = valid_depth_mask(second.points, max_depth)

        def feature_dict(payload, mask):
            return {
                "keypoints": torch.from_numpy(payload.keypoints[mask])[None].cuda(),
                "descriptors": torch.from_numpy(payload.descriptors[mask])[
                    None
                ].cuda(),
                "image_size": torch.from_numpy(payload.image_size)[None].cuda(),
            }

        output = self.matcher(
            {
                "image0": feature_dict(first, first_mask),
                "image1": feature_dict(second, second_mask),
            }
        )
        matches = output["matches"][0]
        if matches.numel() == 0:
            return np.empty((0, 3)), np.empty((0, 3)), 0
        first_indices, second_indices = matches.mT
        first_points = first.points[first_mask][first_indices.cpu().numpy()]
        second_points = second.points[second_mask][second_indices.cpu().numpy()]
        match_count = int(matches.shape[0])
        del output, matches
        torch.cuda.empty_cache()
        return first_points, second_points, match_count


def run_geometric_verification(args) -> Path:
    artifact_root = Path(args.artifact_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifact_root(artifact_root, args.robot_ids)
    print(
        "[GV] loaded "
        + ", ".join(
            f"{robot_id}={artifact.keyframe_count} keyframes"
            for robot_id, artifact in artifacts.items()
        ),
        flush=True,
    )
    raw_base = output_dir / "raw_tracking_graph"
    write_raw_pose_graph(
        artifacts,
        raw_base,
        anchor_robot_id=args.anchor_robot,
        odometry_weight=args.odometry_weight,
    )
    bows = {
        robot_id: compute_bow_vectors(
            artifact,
            Path(args.orb_vocab),
            output_dir / "bow_cache" / f"{robot_id}.npz",
        )
        for robot_id, artifact in artifacts.items()
    }
    extractor = OfflineKeyframeExtractor(
        backend=args.local_feature_backend,
        random_seed=args.random_seed,
    )
    verifier = RobustSim3Verifier(
        noise_bound=args.teaser_noise_bound,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        teaser_required=True,
    )
    constraints = []
    events = []
    pair_diagnostics = {}
    counters = defaultdict(int)
    payload_root = output_dir / "payload_cache"

    for owner_id, remote_id in combinations(sorted(artifacts), 2):
        owner = artifacts[owner_id]
        remote = artifacts[remote_id]
        detector = BowCandidateDetector(
            threshold=args.bow_threshold,
            repetitions=args.bow_repetitions,
            nms_radius=args.bow_nms_radius,
            backfill=False,
            reserve_inflight=True,
        )
        for keyframe_id, bow in bows[owner_id].items():
            detector.add_local(keyframe_id, bow)
        print(
            f"[GV] retrieval pair {owner_id}->{remote_id}: "
            f"{len(bows[owner_id])}x{len(bows[remote_id])}",
            flush=True,
        )
        for remote_keyframe_id, remote_bow in bows[remote_id].items():
            match = detector.observe(
                FrameIdentity(
                    remote.robot_id,
                    remote.session_id,
                    remote_keyframe_id,
                ),
                remote_bow,
            )
            if match is None:
                continue
            counters["top1_candidates"] += 1
            print(
                f"[GV] candidate {owner_id}/{match.local_keyframe_id} <-> "
                f"{remote_id}/{remote_keyframe_id}, score={match.score:.4f}",
                flush=True,
            )
            event = {
                "owner_robot_id": owner_id,
                "owner_keyframe_id": match.local_keyframe_id,
                "remote_robot_id": remote_id,
                "remote_keyframe_id": remote_keyframe_id,
                "retrieval_score": match.score,
                "retrieval_rank": 1,
            }
            try:
                owner_payload = extractor.payload(
                    owner, match.local_keyframe_id, payload_root
                )
                remote_payload = extractor.payload(
                    remote, remote_keyframe_id, payload_root
                )
                owner_points, remote_points, match_count = extractor.match(
                    owner_payload, remote_payload, args.max_depth
                )
                event.update(
                    {
                        "matches": match_count,
                        "owner_depth": depth_statistics(
                            owner_payload.points, args.max_depth
                        ),
                        "remote_depth": depth_statistics(
                            remote_payload.points, args.max_depth
                        ),
                    }
                )
                if match_count < args.min_inliers:
                    event.update(status="rejected", stage="feature_matches")
                    detector.reject(match)
                    counters["rejected_feature_matches"] += 1
                    events.append(event)
                    continue
                result = verifier.verify(owner_points, remote_points)
                event.update(
                    {
                        "inliers": result.inliers,
                        "inlier_ratio": result.inlier_ratio,
                        "scale": result.scale,
                        "verification_method": result.method,
                    }
                )
                if not result.success:
                    event.update(status="rejected", stage="sim3")
                    detector.reject(match)
                    counters["rejected_sim3"] += 1
                    events.append(event)
                    continue
                detector.confirm(match)
                event.update(status="accepted", stage="sim3")
                counters["accepted"] += 1
                weight = max(float(result.inlier_ratio), 0.05) * min(
                    float(result.inliers) / 30.0, 3.0
                )
                constraints.append(
                    RobotMapConstraint(
                        query_robot=(owner.robot_id, owner.session_id),
                        match_robot=(remote.robot_id, remote.session_id),
                        query_pose=Sim3.from_pose(owner_payload.pose),
                        match_pose=Sim3.from_pose(remote_payload.pose),
                        query_to_match=Sim3(
                            result.translation, result.rotation, result.scale
                        ),
                        weight=weight,
                        query_keyframe_id=match.local_keyframe_id,
                        match_keyframe_id=remote_keyframe_id,
                        bow_score=match.score,
                        inliers=result.inliers,
                        inlier_ratio=result.inlier_ratio,
                        verification_method=result.method,
                    )
                )
                print(
                    f"[GV] accepted: matches={match_count}, "
                    f"inliers={result.inliers}, ratio={result.inlier_ratio:.3f}, "
                    f"scale={result.scale:.4f}",
                    flush=True,
                )
                events.append(event)
            except Exception as error:
                detector.reject(match)
                counters["errors"] += 1
                event.update(status="error", stage="payload", error=str(error))
                events.append(event)
                print(f"[GV] candidate error: {error}", flush=True)
            finally:
                torch.cuda.empty_cache()
        pair_diagnostics[f"{owner_id}->{remote_id}"] = (
            detector.diagnostics_snapshot()
        )

    paths = {
        robot_id: [Sim3.from_pose(pose) for pose in artifact.keyframe_poses_xyzw]
        for robot_id, artifact in artifacts.items()
    }
    graph = build_keyframe_graph(
        constraints,
        CentralizedPgoResult({}, True, 0.0, 0.0),
        paths,
        {robot_id: artifact.session_id for robot_id, artifact in artifacts.items()},
        args.anchor_robot,
        odometry_weight=args.odometry_weight,
        align_to_global=False,
        timestamps={
            robot_id: artifact.keyframe_timestamps.tolist()
            for robot_id, artifact in artifacts.items()
        },
    )
    graph.metadata.update(
        {
            "pipeline_stage": "geometric_verification",
            "input_coordinate_frame": "per_robot_local_map_original_scale",
            "input_contains_global_optimization": False,
            "retrieval": "dbow2_top1",
            "retrieval_rank": 1,
            "local_features": args.local_feature_backend,
            "matcher": "lightglue",
            "geometric_verifier": "teaser++",
            "artifact_state_sha256": {
                robot_id: artifact.manifest["state_sha256"]
                for robot_id, artifact in artifacts.items()
            },
            "artifact_keyframe_images_sha256": {
                robot_id: artifact.manifest["keyframe_images_sha256"]
                for robot_id, artifact in artifacts.items()
            },
        }
    )
    graph_base = output_dir / "unoptimized_verified_graph"
    write_json(graph, graph_base.with_suffix(".json"))
    write_g2o(graph, graph_base.with_suffix(".g2o"))
    for robot_id, robot_graph in split_keyframe_graph_by_robot(graph).items():
        robot_base = graph_base.with_name(f"{graph_base.name}_{robot_id}")
        write_json(robot_graph, robot_base.with_suffix(".json"))
        write_g2o(robot_graph, robot_base.with_suffix(".g2o"))
    diagnostics = {
        "format": "dpvo_offline_geometric_verification",
        "version": 1,
        "top1_only": True,
        "parameters": vars(args),
        "counters": dict(counters),
        "accepted_loop_count": len(constraints),
        "pair_diagnostics": pair_diagnostics,
        "events": events,
        "output_graph": str(graph_base.with_suffix(".json")),
    }
    # argparse stores Paths as strings in the CLI, but keep this robust for
    # direct programmatic calls as well.
    diagnostics["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in diagnostics["parameters"].items()
    }
    _atomic_json(output_dir / "verification.json", diagnostics)
    print(
        f"[GV] complete: {len(constraints)} accepted loops from "
        f"{counters['top1_candidates']} top-1 candidates",
        flush=True,
    )
    return graph_base.with_suffix(".json")


def _parser():
    parser = argparse.ArgumentParser(
        description="Staged multi-robot DPVO offline pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble = subparsers.add_parser("assemble", help="write the raw VO graph")
    assemble.add_argument("--artifact-root", required=True)
    assemble.add_argument("--output-base", required=True)
    assemble.add_argument("--anchor-robot", default="robot0")
    assemble.add_argument("--odometry-weight", type=float, default=100.0)
    assemble.add_argument(
        "--robot-ids",
        nargs="+",
        help="Optional exact robot subset; omission loads every artifact.",
    )

    verify = subparsers.add_parser(
        "verify", help="run top-1 BoW, LightGlue, and TEASER++"
    )
    verify.add_argument("--artifact-root", required=True)
    verify.add_argument("--orb-vocab", required=True)
    verify.add_argument("--output-dir", required=True)
    verify.add_argument("--anchor-robot", default="robot0")
    verify.add_argument("--local-feature-backend", default="disk")
    verify.add_argument("--bow-threshold", type=float, default=0.01)
    verify.add_argument("--bow-repetitions", type=int, default=1)
    verify.add_argument("--bow-nms-radius", type=int, default=10)
    verify.add_argument("--teaser-noise-bound", type=float, default=0.10)
    verify.add_argument("--min-inliers", type=int, default=15)
    verify.add_argument("--min-inlier-ratio", type=float, default=0.15)
    verify.add_argument("--max-depth", type=float, default=20.0)
    verify.add_argument("--odometry-weight", type=float, default=100.0)
    verify.add_argument("--random-seed", type=int, default=1234)
    verify.add_argument(
        "--robot-ids",
        nargs="+",
        help="Optional exact robot subset; omission loads every artifact.",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "assemble":
        artifacts = load_artifact_root(Path(args.artifact_root), args.robot_ids)
        write_raw_pose_graph(
            artifacts,
            Path(args.output_base),
            anchor_robot_id=args.anchor_robot,
            odometry_weight=args.odometry_weight,
        )
        return
    graph_path = run_geometric_verification(args)
    print(graph_path)


if __name__ == "__main__":
    main()

"""Transport-neutral distributed loop closure for multi-robot DPVO.

The communication policy is deliberately two-stage:

1. Every stable local keyframe publishes only its place descriptor (MegaLoc or
   the legacy sparse DBoW2 vector).
2. A full XFeat keyframe payload is requested only after repeated top-1 place
   matches, then LighterGlue correspondences are verified with TEASER++.

ROS 2 transport lives in ``ros2/dpvo_multi_robot``. Keeping the matching and
verification logic here makes it testable without a ROS installation.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import threading
import time
from typing import Optional, Protocol

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ..map_gauge import scale_camera_points, transform_pose_xyzw

try:
    from .long_term import LongTermLoopClosure
    _LONG_TERM_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    # The sparse-BoW and TEASER++ primitives are useful in build-time tests
    # before the optional native dpretrieval extension exists.
    LongTermLoopClosure = object
    _LONG_TERM_IMPORT_ERROR = error


@dataclass(frozen=True, order=True)
class FrameIdentity:
    robot_id: str
    session_id: str
    keyframe_id: int


@dataclass
class SparseBow:
    word_ids: np.ndarray
    word_values: np.ndarray

    def __post_init__(self):
        self.word_ids = np.asarray(self.word_ids, dtype=np.uint32).reshape(-1)
        self.word_values = np.asarray(self.word_values, dtype=np.float32).reshape(-1)
        if self.word_ids.shape != self.word_values.shape:
            raise ValueError("BoW word IDs and values must have equal length")
        if self.word_ids.size > 1 and np.any(self.word_ids[1:] <= self.word_ids[:-1]):
            order = np.argsort(self.word_ids)
            self.word_ids = self.word_ids[order]
            self.word_values = self.word_values[order]


@dataclass
class GlobalDescriptor:
    values: np.ndarray
    model_id: str

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if self.values.size == 0:
            raise ValueError("global descriptor must not be empty")
        if not np.isfinite(self.values).all():
            raise ValueError("global descriptor contains non-finite values")
        norm = float(np.linalg.norm(self.values))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("global descriptor has zero norm")
        self.values /= norm
        if not self.model_id:
            raise ValueError("global descriptor model_id must not be empty")


@dataclass
class KeyframePayload:
    frame: FrameIdentity
    timestamp: float
    pose: np.ndarray
    points: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    image_size: np.ndarray


@dataclass(frozen=True)
class BowMatch:
    local_keyframe_id: int
    remote_frame: FrameIdentity
    score: float


@dataclass
class VerificationResult:
    success: bool
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    scale: float = 1.0
    inliers: int = 0
    inlier_ratio: float = 0.0
    method: str = "none"


@dataclass
class InterRobotConstraint:
    query_frame: FrameIdentity
    match_frame: FrameIdentity
    query_pose: np.ndarray
    match_pose: np.ndarray
    bow_score: float
    rotation: np.ndarray
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    scale: float
    inliers: int
    inlier_ratio: float
    verification_method: str


class DistributedTransport(Protocol):
    """Interface implemented by the ROS 2 bridge."""

    def bind(self, backend: "DistributedLongTermLoopClosure") -> None: ...

    def publish_bow(
        self,
        frame: FrameIdentity,
        bow: SparseBow,
        vocabulary_id: str,
        timestamp: float,
    ) -> None: ...

    def publish_global_descriptor(
        self,
        frame: FrameIdentity,
        descriptor: GlobalDescriptor,
        timestamp: float,
    ) -> None: ...

    def request_keyframe(self, match: BowMatch) -> None: ...

    def publish_constraint(self, constraint: InterRobotConstraint) -> None: ...


def vocabulary_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bow_score(first: SparseBow, second: SparseBow) -> float:
    """Compute the DBoW2 L1 score for normalized non-negative BowVectors."""

    _, first_idx, second_idx = np.intersect1d(
        first.word_ids,
        second.word_ids,
        assume_unique=True,
        return_indices=True,
    )
    if first_idx.size == 0:
        return 0.0
    return float(
        np.minimum(first.word_values[first_idx], second.word_values[second_idx]).sum()
    )


def valid_depth_mask(points: np.ndarray, max_depth: float) -> np.ndarray:
    """Select finite positive-depth points with an optional upper bound.

    A non-positive ``max_depth`` disables the upper bound. This is useful for
    monocular maps because their local depth scale is arbitrary until Sim(3)
    alignment.
    """

    points = np.asarray(points)
    finite = np.isfinite(points).all(axis=1)
    mask = finite & (points[:, 2] > 0)
    if max_depth > 0:
        mask &= points[:, 2] < max_depth
    return mask


def depth_statistics(points: np.ndarray, max_depth: float) -> dict:
    points = np.asarray(points)
    finite = np.isfinite(points).all(axis=1)
    positive = finite & (points[:, 2] > 0)
    valid = valid_depth_mask(points, max_depth)
    depths = points[positive, 2].astype(np.float64, copy=False)
    result = {
        "total": int(points.shape[0]),
        "finite": int(np.count_nonzero(finite)),
        "positive_depth": int(np.count_nonzero(positive)),
        "valid_depth": int(np.count_nonzero(valid)),
        "max_depth": float(max_depth),
    }
    if depths.size:
        result["positive_depth_summary"] = {
            "minimum": float(depths.min()),
            "median": float(np.median(depths)),
            "p90": float(np.percentile(depths, 90)),
            "p95": float(np.percentile(depths, 95)),
            "maximum": float(depths.max()),
        }
    return result


class BowCandidateDetector:
    """Find repeated cross-robot BoW matches before requesting keyframes."""

    def __init__(
        self,
        threshold: float,
        repetitions: int,
        nms_radius: int,
        backfill: bool = True,
        reserve_inflight: bool = True,
    ):
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        self.threshold = threshold
        self.repetitions = repetitions
        self.nms_radius = nms_radius
        self.backfill = backfill
        self.reserve_inflight = reserve_inflight
        self.local_bows: OrderedDict[int, SparseBow] = OrderedDict()
        self.inverted_index = defaultdict(dict)
        self.remote_bows = defaultdict(OrderedDict)
        self.remote_inverted_index = defaultdict(lambda: defaultdict(dict))
        self.remote_best = defaultdict(dict)
        self.history = defaultdict(lambda: deque(maxlen=repetitions))
        self.confirmed = defaultdict(list)
        self.reserved = defaultdict(list)
        self.emitted = defaultdict(set)
        self.threshold_frames = set()
        self.observations = 0
        self.threshold_hits = 0
        self.candidates = 0
        self.backfill_candidates = 0
        self.nms_rejections = 0
        self.max_score = 0.0
        self.below_threshold = 0
        self.repetition_waits = 0
        self.sequence_resets = 0
        self.candidates_by_remote = defaultdict(int)
        self.backfill_candidates_by_remote = defaultdict(int)
        self.nms_rejections_by_remote = defaultdict(int)
        self.below_threshold_by_remote = defaultdict(int)
        self.repetition_waits_by_remote = defaultdict(int)
        self.sequence_resets_by_remote = defaultdict(int)

    @staticmethod
    def _remote_key(frame: FrameIdentity) -> tuple[str, str]:
        return frame.robot_id, frame.session_id

    def _store_remote(self, frame: FrameIdentity, bow: SparseBow) -> None:
        remote_key = self._remote_key(frame)
        previous = self.remote_bows[remote_key].get(frame.keyframe_id)
        if previous is not None:
            _, previous_bow = previous
            for word_id in previous_bow.word_ids:
                postings = self.remote_inverted_index[remote_key][int(word_id)]
                postings.pop(frame.keyframe_id, None)
                if not postings:
                    del self.remote_inverted_index[remote_key][int(word_id)]
        self.remote_bows[remote_key][frame.keyframe_id] = (frame, bow)
        for word_id, word_value in zip(
            bow.word_ids, bow.word_values, strict=True
        ):
            self.remote_inverted_index[remote_key][int(word_id)][
                frame.keyframe_id
            ] = float(word_value)

    def _record_threshold_hit(self, match: BowMatch) -> None:
        token = (self._remote_key(match.remote_frame), match.remote_frame.keyframe_id)
        if token not in self.threshold_frames:
            self.threshold_frames.add(token)
            self.threshold_hits += 1

    def _reserve_candidate(self, candidate: BowMatch) -> Optional[BowMatch]:
        remote_key = self._remote_key(candidate.remote_frame)
        pair = (candidate.local_keyframe_id, candidate.remote_frame.keyframe_id)
        if self.reserve_inflight and pair in self.emitted[remote_key]:
            return None
        suppression_points = self.confirmed[remote_key]
        if self.reserve_inflight:
            suppression_points = suppression_points + self.reserved[remote_key]
        for previous_local, previous_remote in suppression_points:
            distance_sq = (
                (candidate.local_keyframe_id - previous_local) ** 2
                + (candidate.remote_frame.keyframe_id - previous_remote) ** 2
            )
            if distance_sq < self.nms_radius**2:
                self.nms_rejections += 1
                self.nms_rejections_by_remote[remote_key] += 1
                return None
        if self.reserve_inflight:
            self.reserved[remote_key].append(pair)
            self.emitted[remote_key].add(pair)
        self.candidates += 1
        self.candidates_by_remote[remote_key] += 1
        return candidate

    def _backfill_remote_matches(
        self,
        keyframe_id: int,
        bow: SparseBow,
    ) -> list[BowMatch]:
        if not self.backfill or not self.remote_bows:
            return []

        candidates = []
        for remote_key, inverted_index in self.remote_inverted_index.items():
            scores = defaultdict(float)
            for word_id, local_value in zip(
                bow.word_ids, bow.word_values, strict=True
            ):
                for remote_id, remote_value in inverted_index.get(
                    int(word_id), {}
                ).items():
                    scores[remote_id] += min(float(local_value), remote_value)

            changed_remote_ids = []
            for remote_id, score in scores.items():
                previous = self.remote_best[remote_key].get(remote_id)
                if previous is not None and (
                    score < previous.score
                    or (
                        np.isclose(score, previous.score)
                        and keyframe_id >= previous.local_keyframe_id
                    )
                ):
                    continue
                remote_frame, _ = self.remote_bows[remote_key][remote_id]
                match = BowMatch(keyframe_id, remote_frame, float(score))
                self.remote_best[remote_key][remote_id] = match
                self.max_score = max(self.max_score, score)
                if score >= self.threshold:
                    self._record_threshold_hit(match)
                    changed_remote_ids.append(remote_id)

            window_starts = {
                remote_id - offset
                for remote_id in changed_remote_ids
                for offset in range(self.repetitions)
            }
            best = self.remote_best[remote_key]
            for start in sorted(window_starts):
                window = [best.get(start + offset) for offset in range(self.repetitions)]
                if any(match is None for match in window):
                    continue
                if any(match.score < self.threshold for match in window):
                    continue
                candidate = self._reserve_candidate(window[self.repetitions // 2])
                if candidate is not None:
                    self.backfill_candidates += 1
                    self.backfill_candidates_by_remote[remote_key] += 1
                    candidates.append(candidate)
        return candidates

    def add_local(self, keyframe_id: int, bow: SparseBow) -> list[BowMatch]:
        previous = self.local_bows.get(keyframe_id)
        if previous is not None:
            for word_id in previous.word_ids:
                postings = self.inverted_index[int(word_id)]
                postings.pop(keyframe_id, None)
                if not postings:
                    del self.inverted_index[int(word_id)]
        self.local_bows[keyframe_id] = bow
        for word_id, word_value in zip(
            bow.word_ids, bow.word_values, strict=True
        ):
            self.inverted_index[int(word_id)][keyframe_id] = float(word_value)
        return self._backfill_remote_matches(keyframe_id, bow)

    def _best_local_match(self, remote_bow: SparseBow) -> tuple[int, float]:
        scores = defaultdict(float)
        for word_id, remote_value in zip(
            remote_bow.word_ids, remote_bow.word_values, strict=True
        ):
            for keyframe_id, local_value in self.inverted_index.get(
                int(word_id), {}
            ).items():
                scores[keyframe_id] += min(local_value, float(remote_value))

        # OrderedDict iteration preserves the previous implementation's
        # deterministic earliest-keyframe tie break, including an all-zero
        # comparison.
        local_id = max(
            self.local_bows,
            key=lambda keyframe_id: scores.get(keyframe_id, 0.0),
        )
        return local_id, float(scores.get(local_id, 0.0))

    def observe(self, frame: FrameIdentity, remote_bow: SparseBow) -> Optional[BowMatch]:
        self.observations += 1
        self._store_remote(frame, remote_bow)
        if not self.local_bows:
            return None

        local_id, score = self._best_local_match(remote_bow)
        match = BowMatch(local_id, frame, score)
        remote_key = self._remote_key(frame)
        self.remote_best[remote_key][frame.keyframe_id] = match
        self.max_score = max(self.max_score, score)
        if score < self.threshold:
            self.below_threshold += 1
            self.below_threshold_by_remote[remote_key] += 1
            self.history[remote_key].clear()
            return None
        self._record_threshold_hit(match)

        history = self.history[remote_key]
        if history and frame.keyframe_id != history[-1].remote_frame.keyframe_id + 1:
            self.sequence_resets += 1
            self.sequence_resets_by_remote[remote_key] += 1
            history.clear()
        history.append(match)
        if len(history) < self.repetitions:
            self.repetition_waits += 1
            self.repetition_waits_by_remote[remote_key] += 1
            return None

        first, last = history[0], history[-1]
        if last.remote_frame.keyframe_id - first.remote_frame.keyframe_id != self.repetitions - 1:
            return None

        candidate = history[len(history) // 2]
        return self._reserve_candidate(candidate)

    @staticmethod
    def _score_summary(scores: list[float]) -> dict:
        if not scores:
            return {"count": 0}
        values = np.asarray(scores, dtype=np.float64)
        return {
            "count": int(values.size),
            "minimum": float(values.min()),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "maximum": float(values.max()),
        }

    def diagnostics_snapshot(self) -> dict:
        remote_keys = sorted(
            set(self.remote_bows)
            | set(self.remote_best)
            | set(self.candidates_by_remote)
            | set(self.nms_rejections_by_remote)
        )
        by_remote = {}
        for remote_key in remote_keys:
            scores = [
                float(match.score)
                for match in self.remote_best[remote_key].values()
            ]
            threshold_hits = sum(
                1
                for key, _keyframe_id in self.threshold_frames
                if key == remote_key
            )
            label = f"{remote_key[0]}:{remote_key[1]}"
            by_remote[label] = {
                "robot_id": remote_key[0],
                "session_id": remote_key[1],
                "observed_keyframes": len(self.remote_bows[remote_key]),
                "top1_score": self._score_summary(scores),
                "threshold_hits": threshold_hits,
                "below_threshold_observations": self.below_threshold_by_remote[
                    remote_key
                ],
                "repetition_waits": self.repetition_waits_by_remote[remote_key],
                "sequence_resets": self.sequence_resets_by_remote[remote_key],
                "candidates": self.candidates_by_remote[remote_key],
                "backfill_candidates": self.backfill_candidates_by_remote[
                    remote_key
                ],
                "nms_rejections": self.nms_rejections_by_remote[remote_key],
                "confirmed": len(self.confirmed[remote_key]),
                "reserved": len(self.reserved[remote_key]),
                "emitted_pairs": len(self.emitted[remote_key]),
                "history_length": len(self.history[remote_key]),
            }
        return {
            "parameters": {
                "threshold": self.threshold,
                "repetitions": self.repetitions,
                "nms_radius": self.nms_radius,
                "backfill": self.backfill,
                "reserve_inflight": self.reserve_inflight,
            },
            "observations": self.observations,
            "threshold_hits": self.threshold_hits,
            "below_threshold_observations": self.below_threshold,
            "repetition_waits": self.repetition_waits,
            "sequence_resets": self.sequence_resets,
            "candidates": self.candidates,
            "backfill_candidates": self.backfill_candidates,
            "nms_rejections": self.nms_rejections,
            "max_score": self.max_score,
            "local_keyframes": len(self.local_bows),
            "by_remote": by_remote,
        }

    def confirm(self, match: BowMatch) -> None:
        remote_key = (match.remote_frame.robot_id, match.remote_frame.session_id)
        pair = (match.local_keyframe_id, match.remote_frame.keyframe_id)
        if pair in self.reserved[remote_key]:
            self.reserved[remote_key].remove(pair)
        self.confirmed[remote_key].append(pair)
        self.history[remote_key].clear()

    def reject(self, match: BowMatch) -> None:
        """Release an NMS reservation after detail retrieval or verification fails."""

        if not self.reserve_inflight:
            # Reproduce the original online scheduler: keep the rolling
            # retrieval history so consecutive nearby frames may be queued
            # while geometric verification is still in flight.
            return

        remote_key = (match.remote_frame.robot_id, match.remote_frame.session_id)
        pair = (match.local_keyframe_id, match.remote_frame.keyframe_id)
        if pair in self.reserved[remote_key]:
            self.reserved[remote_key].remove(pair)
        self.history[remote_key].clear()


class GlobalDescriptorCandidateDetector(BowCandidateDetector):
    """Top-1 cosine retrieval with the same repetition and NMS gates as BoW."""

    def __init__(
        self,
        threshold: float,
        repetitions: int,
        nms_radius: int,
        backfill: bool = True,
        reserve_inflight: bool = True,
    ):
        super().__init__(
            threshold,
            repetitions,
            nms_radius,
            backfill,
            reserve_inflight,
        )
        # Preserve the inherited diagnostics/state machinery while exposing
        # names that describe what is actually stored.
        self.local_descriptors = self.local_bows
        self.remote_descriptors = self.remote_bows

    def _store_remote(
        self,
        frame: FrameIdentity,
        descriptor: GlobalDescriptor,
    ) -> None:
        remote_key = self._remote_key(frame)
        self.remote_descriptors[remote_key][frame.keyframe_id] = (
            frame,
            descriptor,
        )

    def _best_local_match(
        self,
        remote_descriptor: GlobalDescriptor,
    ) -> tuple[int, float]:
        if not self.local_descriptors:
            raise ValueError("no local descriptors are available")
        local_id, local_descriptor = max(
            self.local_descriptors.items(),
            key=lambda item: float(item[1].values @ remote_descriptor.values),
        )
        return local_id, float(local_descriptor.values @ remote_descriptor.values)

    def _backfill_remote_matches(
        self,
        keyframe_id: int,
        descriptor: GlobalDescriptor,
    ) -> list[BowMatch]:
        if not self.backfill or not self.remote_descriptors:
            return []

        candidates = []
        for remote_key, descriptors in self.remote_descriptors.items():
            changed_remote_ids = []
            for remote_id, (remote_frame, remote_descriptor) in descriptors.items():
                score = float(descriptor.values @ remote_descriptor.values)
                previous = self.remote_best[remote_key].get(remote_id)
                if previous is not None and (
                    score < previous.score
                    or (
                        np.isclose(score, previous.score)
                        and keyframe_id >= previous.local_keyframe_id
                    )
                ):
                    continue
                match = BowMatch(keyframe_id, remote_frame, score)
                self.remote_best[remote_key][remote_id] = match
                self.max_score = max(self.max_score, score)
                if score >= self.threshold:
                    self._record_threshold_hit(match)
                    changed_remote_ids.append(remote_id)

            window_starts = {
                remote_id - offset
                for remote_id in changed_remote_ids
                for offset in range(self.repetitions)
            }
            best = self.remote_best[remote_key]
            for start in sorted(window_starts):
                window = [
                    best.get(start + offset)
                    for offset in range(self.repetitions)
                ]
                if any(match is None for match in window):
                    continue
                if any(match.score < self.threshold for match in window):
                    continue
                candidate = self._reserve_candidate(
                    window[self.repetitions // 2]
                )
                if candidate is not None:
                    self.backfill_candidates += 1
                    self.backfill_candidates_by_remote[remote_key] += 1
                    candidates.append(candidate)
        return candidates

    def add_local(
        self,
        keyframe_id: int,
        descriptor: GlobalDescriptor,
    ) -> list[BowMatch]:
        self.local_descriptors[keyframe_id] = descriptor
        return self._backfill_remote_matches(keyframe_id, descriptor)

    def observe(
        self,
        frame: FrameIdentity,
        remote_descriptor: GlobalDescriptor,
    ) -> Optional[BowMatch]:
        return super().observe(frame, remote_descriptor)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.T @ src_centered / src.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.square(src_centered).sum() / src.shape[0]
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("degenerate point set")
    scale = np.trace(np.diag(singular) @ correction) / variance
    translation = dst_mean - scale * rotation @ src_mean
    return rotation, translation, float(scale)


class RobustSim3Verifier:
    """Estimate Sim(3) with TEASER++, with a deterministic RANSAC fallback."""

    def __init__(
        self,
        noise_bound: float,
        min_inliers: int,
        min_inlier_ratio: float,
        teaser_required: bool = False,
        ransac_iterations: int = 400,
    ):
        self.noise_bound = noise_bound
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.teaser_required = teaser_required
        self.ransac_iterations = ransac_iterations

    def verify(self, src: np.ndarray, dst: np.ndarray) -> VerificationResult:
        src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
        if src.shape != dst.shape or src.shape[0] < 3:
            return VerificationResult(False)

        try:
            module = importlib.import_module("teaserpp_python")
        except ImportError:
            if self.teaser_required:
                return VerificationResult(False, method="teaser++ unavailable")
            return self._ransac(src, dst)

        try:
            result = self._teaser(module, src, dst)
        except Exception as error:
            if self.teaser_required:
                return VerificationResult(False, method=f"teaser++ error: {error}")
            result = self._ransac(src, dst)
            result.method += " (TEASER++ failed)"
            return result
        return result

    def _teaser(self, module, src: np.ndarray, dst: np.ndarray) -> VerificationResult:
        solver_type = module.RobustRegistrationSolver
        params = solver_type.Params()
        params.cbar2 = 1.0
        params.noise_bound = self.noise_bound
        params.estimate_scaling = True
        params.rotation_estimation_algorithm = (
            solver_type.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        params.rotation_gnc_factor = 1.4
        params.rotation_max_iterations = 100
        params.rotation_cost_threshold = 1e-12
        solver = solver_type(params)

        # TEASER++ consumes 3xN Eigen column-major matrices. Fortran layout also
        # avoids a known NumPy/pybind layout pitfall in older Python bindings.
        solver.solve(
            np.asfortranarray(src.T),
            np.asfortranarray(dst.T),
        )
        solution = solver.getSolution()
        if hasattr(solution, "valid") and not solution.valid:
            return VerificationResult(False, method="teaser++")
        return self._evaluate(
            src,
            dst,
            np.asarray(solution.rotation, dtype=np.float64),
            np.asarray(solution.translation, dtype=np.float64).reshape(3),
            float(solution.scale),
            "teaser++",
        )

    def _ransac(self, src: np.ndarray, dst: np.ndarray) -> VerificationResult:
        rng = np.random.default_rng(0)
        best = None
        best_mask = None
        for _ in range(self.ransac_iterations):
            sample = rng.choice(src.shape[0], 3, replace=False)
            try:
                rotation, translation, scale = _umeyama(src[sample], dst[sample])
            except (ValueError, np.linalg.LinAlgError):
                continue
            residual = np.linalg.norm(
                scale * (src @ rotation.T) + translation - dst,
                axis=1,
            )
            mask = residual < self.noise_bound
            if best_mask is None or mask.sum() > best_mask.sum():
                best = (rotation, translation, scale)
                best_mask = mask

        if best_mask is None or best_mask.sum() < 3:
            return VerificationResult(False, method="ransac")
        try:
            best = _umeyama(src[best_mask], dst[best_mask])
        except (ValueError, np.linalg.LinAlgError):
            return VerificationResult(False, method="ransac")
        return self._evaluate(src, dst, *best, method="ransac")

    def _evaluate(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        scale: float,
        method: str,
    ) -> VerificationResult:
        residual = np.linalg.norm(
            scale * (src @ rotation.T) + translation - dst,
            axis=1,
        )
        inliers = int(np.count_nonzero(residual < self.noise_bound))
        ratio = inliers / src.shape[0]
        valid = (
            np.isfinite(scale)
            and scale > 0
            and inliers >= self.min_inliers
            and ratio >= self.min_inlier_ratio
        )
        return VerificationResult(
            valid,
            rotation=rotation,
            translation=translation,
            scale=scale,
            inliers=inliers,
            inlier_ratio=ratio,
            method=method,
        )


class DistributedLongTermLoopClosure(LongTermLoopClosure):
    """Classic DPVO loop closure plus cross-robot ROS transport hooks."""

    def __init__(
        self,
        cfg,
        patchgraph,
        transport: DistributedTransport,
        robot_id: str,
        session_id: str,
        vocabulary_id: Optional[str] = None,
        distributed_enabled: bool = True,
    ):
        if _LONG_TERM_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "Distributed DPVO requires the classic dpretrieval backend"
            ) from _LONG_TERM_IMPORT_ERROR
        self.transport = transport
        self.robot_id = robot_id
        self.session_id = session_id
        self.distributed_enabled = bool(distributed_enabled)
        self.retrieval_backend = cfg.MULTI_ROBOT_RETRIEVAL_BACKEND.lower()
        if self.retrieval_backend not in ("dbow2", "megaloc"):
            raise ValueError(
                "MULTI_ROBOT_RETRIEVAL_BACKEND must be 'dbow2' or 'megaloc', "
                f"got {self.retrieval_backend!r}"
            )
        self.vocabulary_id = vocabulary_id or vocabulary_fingerprint(
            cfg.ORB_VOCAB_PATH
        )
        self.retrieval_model_id = cfg.MULTI_ROBOT_MEGALOC_MODEL_ID
        self.retrieval_descriptor_dimension = (
            cfg.MULTI_ROBOT_MEGALOC_DIMENSION
        )
        if self.retrieval_backend == "megaloc":
            self.candidate_detector = GlobalDescriptorCandidateDetector(
                threshold=cfg.MULTI_ROBOT_MEGALOC_THRESHOLD,
                repetitions=cfg.MULTI_ROBOT_BOW_REPETITIONS,
                nms_radius=cfg.MULTI_ROBOT_BOW_NMS,
                backfill=cfg.MULTI_ROBOT_BOW_BACKFILL,
                reserve_inflight=cfg.MULTI_ROBOT_RESERVE_INFLIGHT,
            )
        else:
            self.candidate_detector = BowCandidateDetector(
                threshold=cfg.LOOP_RETR_THRESH,
                repetitions=cfg.MULTI_ROBOT_BOW_REPETITIONS,
                nms_radius=cfg.MULTI_ROBOT_BOW_NMS,
                backfill=cfg.MULTI_ROBOT_BOW_BACKFILL,
                reserve_inflight=cfg.MULTI_ROBOT_RESERVE_INFLIGHT,
            )
        self.verifier = RobustSim3Verifier(
            noise_bound=cfg.MULTI_ROBOT_TEASER_NOISE_BOUND,
            min_inliers=cfg.MULTI_ROBOT_MIN_INLIERS,
            min_inlier_ratio=cfg.MULTI_ROBOT_MIN_INLIER_RATIO,
            teaser_required=cfg.MULTI_ROBOT_TEASER_REQUIRED,
        )
        self.payload_cache = OrderedDict()
        self.pending_matches = set()
        self.candidate_queue = deque()
        self.delayed_request_queue = deque()
        self.response_queue = deque()
        self.request_attempts = defaultdict(int)
        self.distributed_lock = threading.RLock()
        self.inter_robot_lc_count = 0
        self.diagnostics = defaultdict(int)
        self.candidate_events = []
        self.verification_events = []

        super().__init__(cfg, patchgraph, bow_callback=self._publish_local_retrieval)
        self.global_descriptor_extractor = None
        if self.retrieval_backend == "megaloc":
            from .learned_frontend import MegaLocDescriptorExtractor

            self.global_descriptor_extractor = MegaLocDescriptorExtractor(
                cfg.MULTI_ROBOT_MEGALOC_REPO,
                device=cfg.MULTI_ROBOT_MEGALOC_DEVICE,
            )
        self.transport.bind(self)

    def _diagnostic(self, message: str) -> None:
        logger = getattr(self.transport, "log", None)
        if logger is not None:
            logger(message)
        else:
            print(message, flush=True)

    def _publish_local_retrieval(self, keyframe_id: int, entries) -> None:
        if not self.distributed_enabled:
            return
        if self.retrieval_backend == "megaloc":
            # Detailed geometry triangulates the (i-1, i, i+1) triplet, so the
            # first frame cannot ever satisfy an on-demand keyframe request.
            if keyframe_id < 1:
                return
            image = self.imcache.load_frames(
                [keyframe_id],
                self.pg.intrinsics.device,
            )
            values = (
                self.global_descriptor_extractor(image)[0]
                .detach()
                .cpu()
                .numpy()
            )
            descriptor = GlobalDescriptor(values, self.retrieval_model_id)
            if descriptor.values.size != self.retrieval_descriptor_dimension:
                raise RuntimeError(
                    "MegaLoc descriptor dimension mismatch: "
                    f"expected {self.retrieval_descriptor_dimension}, "
                    f"got {descriptor.values.size}"
                )
            with self.distributed_lock:
                matches = self.candidate_detector.add_local(
                    keyframe_id,
                    descriptor,
                )
                for match in matches:
                    self._queue_match(match, source="backfill")
            timestamp = float(self.pg.tstamps_[keyframe_id])
            self.transport.publish_global_descriptor(
                FrameIdentity(self.robot_id, self.session_id, keyframe_id),
                descriptor,
                timestamp,
            )
            return

        if entries:
            word_ids, word_values = zip(*entries)
        else:
            word_ids, word_values = (), ()
        bow = SparseBow(word_ids, word_values)
        with self.distributed_lock:
            matches = self.candidate_detector.add_local(keyframe_id, bow)
            for match in matches:
                self._queue_match(match, source="backfill")
        timestamp = float(self.pg.tstamps_[keyframe_id])
        self.transport.publish_bow(
            FrameIdentity(self.robot_id, self.session_id, keyframe_id),
            bow,
            self.vocabulary_id,
            timestamp,
        )

    # Compatibility for external code that called the old callback directly.
    def _publish_local_bow(self, keyframe_id: int, entries) -> None:
        self._publish_local_retrieval(keyframe_id, entries)

    def _queue_match(self, match: BowMatch, source: str) -> None:
        key = (match.local_keyframe_id, match.remote_frame)
        if key in self.pending_matches:
            return
        self.pending_matches.add(key)
        self.candidate_queue.append(match)
        self.diagnostics["candidates_queued"] += 1
        self.diagnostics[f"candidates_queued_{source}"] += 1
        self.candidate_events.append(
            {
                "source": source,
                "local_keyframe_id": match.local_keyframe_id,
                "remote_robot_id": match.remote_frame.robot_id,
                "remote_session_id": match.remote_frame.session_id,
                "remote_keyframe_id": match.remote_frame.keyframe_id,
                "retrieval_backend": self.retrieval_backend,
                "retrieval_score": match.score,
                "bow_score": match.score,
            }
        )
        self._diagnostic(
            "INTER-ROBOT RETRIEVAL CANDIDATE: "
            f"backend={self.retrieval_backend} "
            f"source={source} local={match.local_keyframe_id} "
            f"remote={match.remote_frame.robot_id}/"
            f"{match.remote_frame.keyframe_id} score={match.score:.6f}"
        )

    def receive_bow(
        self,
        frame: FrameIdentity,
        bow: SparseBow,
        vocabulary_id: str,
    ) -> None:
        if not self.distributed_enabled:
            return
        if self.retrieval_backend != "dbow2":
            return
        if (frame.robot_id, frame.session_id) == (self.robot_id, self.session_id):
            return
        if vocabulary_id != self.vocabulary_id:
            return

        # Deterministic ownership prevents both robots requesting and publishing
        # the same symmetric inter-robot constraint.
        if (self.robot_id, self.session_id) >= (frame.robot_id, frame.session_id):
            return

        with self.distributed_lock:
            match = self.candidate_detector.observe(frame, bow)
            if match is None:
                return
            self._queue_match(match, source="online")

    def receive_global_descriptor(
        self,
        frame: FrameIdentity,
        descriptor: GlobalDescriptor,
    ) -> None:
        if not self.distributed_enabled:
            return
        if self.retrieval_backend != "megaloc":
            return
        if (frame.robot_id, frame.session_id) == (self.robot_id, self.session_id):
            return
        if descriptor.model_id != self.retrieval_model_id:
            self.diagnostics["rejected_retrieval_model"] += 1
            return
        if descriptor.values.size != self.retrieval_descriptor_dimension:
            self.diagnostics["rejected_retrieval_dimension"] += 1
            return

        # Deterministic ownership prevents both robots requesting and publishing
        # the same symmetric inter-robot constraint.
        if (self.robot_id, self.session_id) >= (frame.robot_id, frame.session_id):
            return

        with self.distributed_lock:
            match = self.candidate_detector.observe(frame, descriptor)
            if match is None:
                return
            self._queue_match(match, source="online")

    def receive_keyframe(
        self,
        match: BowMatch,
        payload: Optional[KeyframePayload],
        error: str = "",
    ) -> None:
        with self.distributed_lock:
            self.response_queue.append((match, payload, error))
            self.diagnostics["payload_responses"] += 1
            if error or payload is None:
                self.diagnostics["payload_errors"] += 1

    def export_keyframe(self, frame: FrameIdentity) -> KeyframePayload:
        if (frame.robot_id, frame.session_id) != (self.robot_id, self.session_id):
            raise KeyError("keyframe request is for a different robot session")
        return self._local_payload(frame.keyframe_id)

    def _local_payload(self, keyframe_id: int) -> KeyframePayload:
        if keyframe_id in self.payload_cache:
            payload = self.payload_cache.pop(keyframe_id)
            self.payload_cache[keyframe_id] = payload
            return payload

        from ..lietorch import SE3

        points, features = self.estimate_3d_keypoints(keyframe_id)
        gauge = self.pg.session_from_map_.copy()
        pose = (
            SE3(self.pg.poses_[keyframe_id])
            .inv()
            .data.detach()
            .float()
            .cpu()
            .numpy()
        )
        payload = KeyframePayload(
            frame=FrameIdentity(self.robot_id, self.session_id, keyframe_id),
            timestamp=float(self.pg.tstamps_[keyframe_id]),
            pose=transform_pose_xyzw(gauge, pose).astype(np.float32),
            points=scale_camera_points(
                gauge,
                points.detach().float().cpu().numpy(),
            ).astype(np.float32),
            keypoints=features["keypoints"].squeeze(0).detach().float().cpu().numpy(),
            descriptors=features["descriptors"].squeeze(0).detach().float().cpu().numpy(),
            image_size=features["image_size"].squeeze(0).detach().float().cpu().numpy(),
        )
        self.payload_cache[keyframe_id] = payload
        while len(self.payload_cache) > self.cfg.MULTI_ROBOT_KEYFRAME_CACHE_SIZE:
            self.payload_cache.popitem(last=False)
        return payload

    def attempt_loop_closure(self, n):
        super().attempt_loop_closure(n)
        self.process_transport()

    def process_transport(self):
        """Advance queued distributed work independently of new images."""

        if not self.distributed_enabled:
            return

        if self.lc_in_progress:
            return

        response = None
        request = None
        with self.distributed_lock:
            if self.response_queue:
                response = self.response_queue.popleft()
            elif (
                self.delayed_request_queue
                and self.delayed_request_queue[0][0] <= time.monotonic()
            ):
                _, request = self.delayed_request_queue.popleft()
            elif self.candidate_queue:
                request = self.candidate_queue.popleft()

        if response is not None:
            match, payload, error = response
            key = (match.local_keyframe_id, match.remote_frame)
            accepted = False
            retry_error = error or ("empty keyframe payload" if payload is None else "")
            try:
                if not retry_error:
                    try:
                        accepted = self._verify_remote_match(match, payload)
                    except (FileNotFoundError, KeyError) as cache_error:
                        # A peer can respond while the query's neighboring
                        # images are crossing the stable-cache boundary. The
                        # candidate may recur; it must not terminate odometry.
                        self.diagnostics["payload_cache_misses"] += 1
                        retry_error = str(cache_error)
            finally:
                with self.distributed_lock:
                    attempts = self.request_attempts.get(key, 1)
                    max_attempts = self.cfg.MULTI_ROBOT_KEYFRAME_MAX_ATTEMPTS
                    if retry_error and attempts < max_attempts:
                        self.delayed_request_queue.append(
                            (
                                time.monotonic()
                                + self.cfg.MULTI_ROBOT_KEYFRAME_RETRY_DELAY,
                                match,
                            )
                        )
                        self.diagnostics["keyframe_retries_scheduled"] += 1
                        self._diagnostic(
                            "INTER-ROBOT PAYLOAD RETRY: "
                            f"local={match.local_keyframe_id} "
                            f"remote={match.remote_frame.robot_id}/"
                            f"{match.remote_frame.keyframe_id} "
                            f"attempt={attempts}/{max_attempts} "
                            f"error={retry_error}"
                        )
                        return

                    if retry_error:
                        self.diagnostics["keyframe_retries_exhausted"] += 1
                        self._diagnostic(
                            "INTER-ROBOT PAYLOAD REJECT: "
                            f"local={match.local_keyframe_id} "
                            f"remote={match.remote_frame.robot_id}/"
                            f"{match.remote_frame.keyframe_id} "
                            f"attempts={attempts} error={retry_error}"
                        )
                    if not accepted:
                        self.candidate_detector.reject(match)
                    self.pending_matches.discard(key)
                    self.request_attempts.pop(key, None)
        elif request is not None:
            key = (request.local_keyframe_id, request.remote_frame)
            self.request_attempts[key] += 1
            self.diagnostics["keyframe_requests"] += 1
            self.transport.request_keyframe(request)

    @torch.no_grad()
    def _verify_remote_match(
        self,
        match: BowMatch,
        remote: KeyframePayload,
    ) -> bool:
        self.diagnostics["verification_attempts"] += 1
        local = self._local_payload(match.local_keyframe_id)
        max_depth = self.cfg.MULTI_ROBOT_MAX_DEPTH
        local_mask = valid_depth_mask(local.points, max_depth)
        remote_mask = valid_depth_mask(remote.points, max_depth)
        local_depth = depth_statistics(local.points, max_depth)
        remote_depth = depth_statistics(remote.points, max_depth)
        if (
            np.count_nonzero(local_mask) < self.cfg.MULTI_ROBOT_MIN_INLIERS
            or np.count_nonzero(remote_mask) < self.cfg.MULTI_ROBOT_MIN_INLIERS
        ):
            self.diagnostics["rejected_3d_support"] += 1
            self._record_verification(
                match,
                remote,
                status="rejected",
                stage="3d_support",
                local_depth=local_depth,
                remote_depth=remote_depth,
            )
            self._diagnostic(
                "INTER-ROBOT VERIFY REJECT: "
                f"stage=3d_support local={match.local_keyframe_id} "
                f"remote={remote.frame.robot_id}/{remote.frame.keyframe_id} "
                f"local_points={np.count_nonzero(local_mask)} "
                f"remote_points={np.count_nonzero(remote_mask)}"
            )
            return False

        def feature_dict(payload, mask):
            return {
                "keypoints": torch.from_numpy(payload.keypoints[mask])[None].cuda(),
                "descriptors": torch.from_numpy(payload.descriptors[mask])[None].cuda(),
                "image_size": torch.from_numpy(payload.image_size)[None].cuda(),
            }

        output = self.matcher(
            {
                "image0": feature_dict(local, local_mask),
                "image1": feature_dict(remote, remote_mask),
            }
        )
        matches = output["matches"][0]
        if matches.shape[0] < self.cfg.MULTI_ROBOT_MIN_INLIERS:
            self.diagnostics["rejected_feature_matches"] += 1
            self._record_verification(
                match,
                remote,
                status="rejected",
                stage="feature_matches",
                matches=int(matches.shape[0]),
                local_depth=local_depth,
                remote_depth=remote_depth,
            )
            self._diagnostic(
                "INTER-ROBOT VERIFY REJECT: "
                f"stage=feature_matches local={match.local_keyframe_id} "
                f"remote={remote.frame.robot_id}/{remote.frame.keyframe_id} "
                f"matches={matches.shape[0]}"
            )
            return False
        local_indices, remote_indices = matches.mT
        local_points = local.points[local_mask][local_indices.cpu().numpy()]
        remote_points = remote.points[remote_mask][remote_indices.cpu().numpy()]
        result = self.verifier.verify(local_points, remote_points)
        if not result.success:
            self.diagnostics["rejected_sim3"] += 1
            self._record_verification(
                match,
                remote,
                status="rejected",
                stage="sim3",
                matches=int(matches.shape[0]),
                inliers=result.inliers,
                inlier_ratio=result.inlier_ratio,
                method=result.method,
                local_depth=local_depth,
                remote_depth=remote_depth,
            )
            self._diagnostic(
                "INTER-ROBOT VERIFY REJECT: "
                f"stage=sim3 local={match.local_keyframe_id} "
                f"remote={remote.frame.robot_id}/{remote.frame.keyframe_id} "
                f"matches={matches.shape[0]} inliers={result.inliers} "
                f"ratio={result.inlier_ratio:.3f} method={result.method}"
            )
            return False

        constraint = InterRobotConstraint(
            query_frame=local.frame,
            match_frame=remote.frame,
            query_pose=local.pose,
            match_pose=remote.pose,
            bow_score=match.score,
            rotation=result.rotation,
            translation=result.translation,
            # TEASER exposes read-only buffers; older SciPy requires writable input.
            quaternion_xyzw=Rotation.from_matrix(
                np.array(result.rotation, dtype=np.float64, copy=True)
            ).as_quat(),
            scale=result.scale,
            inliers=result.inliers,
            inlier_ratio=result.inlier_ratio,
            verification_method=result.method,
        )
        self.transport.publish_constraint(constraint)
        with self.distributed_lock:
            self.candidate_detector.confirm(match)
        self.inter_robot_lc_count += 1
        self.diagnostics["accepted"] += 1
        self._record_verification(
            match,
            remote,
            status="accepted",
            stage="sim3",
            matches=int(matches.shape[0]),
            inliers=result.inliers,
            inlier_ratio=result.inlier_ratio,
            scale=result.scale,
            method=result.method,
            local_depth=local_depth,
            remote_depth=remote_depth,
        )
        self._diagnostic(
            "INTER-ROBOT VERIFY ACCEPT: "
            f"local={match.local_keyframe_id} "
            f"remote={remote.frame.robot_id}/{remote.frame.keyframe_id} "
            f"score={match.score:.6f} matches={matches.shape[0]} "
            f"inliers={result.inliers} ratio={result.inlier_ratio:.3f} "
            f"scale={result.scale:.6f} method={result.method}"
        )
        return True

    def _record_verification(
        self,
        match: BowMatch,
        remote: KeyframePayload,
        *,
        status: str,
        stage: str,
        **details,
    ) -> None:
        self.verification_events.append(
            {
                "status": status,
                "stage": stage,
                "local_keyframe_id": match.local_keyframe_id,
                "remote_robot_id": remote.frame.robot_id,
                "remote_session_id": remote.frame.session_id,
                "remote_keyframe_id": remote.frame.keyframe_id,
                "retrieval_score": match.score,
                "bow_score": match.score,
                **details,
            }
        )

    def diagnostics_snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "robot_id": self.robot_id,
            "session_id": self.session_id,
            "front_end": {
                "distributed_enabled": self.distributed_enabled,
                "retrieval": self.retrieval_backend,
                "retrieval_model": (
                    self.retrieval_model_id
                    if self.retrieval_backend == "megaloc"
                    else self.vocabulary_id
                ),
                "local_features": self.cfg.MULTI_ROBOT_LOCAL_FEATURE_BACKEND,
                "matcher": (
                    "xfeat-lighterglue"
                    if self.cfg.MULTI_ROBOT_LOCAL_FEATURE_BACKEND.lower()
                    == "xfeat"
                    else "lightglue-disk"
                ),
                "geometric_verifier": "teaser++",
            },
            "retrieval": self.candidate_detector.diagnostics_snapshot(),
            "pipeline": dict(self.diagnostics),
            "inter_robot_loop_count": self.inter_robot_lc_count,
            "queues": {
                "pending_matches": len(self.pending_matches),
                "candidate_queue": len(self.candidate_queue),
                "delayed_request_queue": len(self.delayed_request_queue),
                "response_queue": len(self.response_queue),
            },
            "candidate_events": list(self.candidate_events),
            "verification_events": list(self.verification_events),
        }

    def terminate(self, n):
        super().terminate(n)
        retrieval_diagnostics = self.candidate_detector.diagnostics_snapshot()
        self._diagnostic(
            f"INTER-ROBOT RETRIEVAL DIAGNOSTICS: {retrieval_diagnostics}"
        )
        self._diagnostic(
            f"INTER-ROBOT VERIFICATION DIAGNOSTICS: {dict(self.diagnostics)}"
        )
        self._diagnostic(f"INTER-ROBOT LC COUNT: {self.inter_robot_lc_count}")

"""Transport-neutral distributed classic loop closure for multi-robot DPVO.

The communication policy is deliberately two-stage:

1. Every stable local keyframe publishes only its sparse DBoW2 vector.
2. A full keyframe payload is requested only after repeated BoW matches.

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


class BowCandidateDetector:
    """Find repeated cross-robot BoW matches before requesting keyframes."""

    def __init__(self, threshold: float, repetitions: int, nms_radius: int):
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        self.threshold = threshold
        self.repetitions = repetitions
        self.nms_radius = nms_radius
        self.local_bows: OrderedDict[int, SparseBow] = OrderedDict()
        self.history = defaultdict(lambda: deque(maxlen=repetitions))
        self.confirmed = defaultdict(list)

    def add_local(self, keyframe_id: int, bow: SparseBow) -> None:
        self.local_bows[keyframe_id] = bow

    def observe(self, frame: FrameIdentity, remote_bow: SparseBow) -> Optional[BowMatch]:
        if not self.local_bows:
            return None

        local_id, score = max(
            (
                (keyframe_id, bow_score(local_bow, remote_bow))
                for keyframe_id, local_bow in self.local_bows.items()
            ),
            key=lambda item: item[1],
        )
        if score < self.threshold:
            self.history[(frame.robot_id, frame.session_id)].clear()
            return None

        remote_key = (frame.robot_id, frame.session_id)
        history = self.history[remote_key]
        if history and frame.keyframe_id != history[-1].remote_frame.keyframe_id + 1:
            history.clear()
        history.append(BowMatch(local_id, frame, score))
        if len(history) < self.repetitions:
            return None

        first, last = history[0], history[-1]
        if last.remote_frame.keyframe_id - first.remote_frame.keyframe_id != self.repetitions - 1:
            return None

        candidate = history[len(history) // 2]
        for previous_local, previous_remote in self.confirmed[remote_key]:
            distance_sq = (
                (candidate.local_keyframe_id - previous_local) ** 2
                + (candidate.remote_frame.keyframe_id - previous_remote) ** 2
            )
            if distance_sq < self.nms_radius**2:
                return None
        return candidate

    def confirm(self, match: BowMatch) -> None:
        remote_key = (match.remote_frame.robot_id, match.remote_frame.session_id)
        self.confirmed[remote_key].append(
            (match.local_keyframe_id, match.remote_frame.keyframe_id)
        )
        self.history[remote_key].clear()


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
    ):
        if _LONG_TERM_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "Distributed DPVO requires the classic dpretrieval backend"
            ) from _LONG_TERM_IMPORT_ERROR
        self.transport = transport
        self.robot_id = robot_id
        self.session_id = session_id
        self.vocabulary_id = vocabulary_id or vocabulary_fingerprint(
            cfg.ORB_VOCAB_PATH
        )
        self.candidate_detector = BowCandidateDetector(
            threshold=cfg.LOOP_RETR_THRESH,
            repetitions=cfg.MULTI_ROBOT_BOW_REPETITIONS,
            nms_radius=cfg.MULTI_ROBOT_BOW_NMS,
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
        self.response_queue = deque()
        self.distributed_lock = threading.RLock()
        self.inter_robot_lc_count = 0

        super().__init__(cfg, patchgraph, bow_callback=self._publish_local_bow)
        self.transport.bind(self)

    def _publish_local_bow(self, keyframe_id: int, entries) -> None:
        if entries:
            word_ids, word_values = zip(*entries)
        else:
            word_ids, word_values = (), ()
        bow = SparseBow(word_ids, word_values)
        with self.distributed_lock:
            self.candidate_detector.add_local(keyframe_id, bow)
        timestamp = float(self.pg.tstamps_[keyframe_id])
        self.transport.publish_bow(
            FrameIdentity(self.robot_id, self.session_id, keyframe_id),
            bow,
            self.vocabulary_id,
            timestamp,
        )

    def receive_bow(
        self,
        frame: FrameIdentity,
        bow: SparseBow,
        vocabulary_id: str,
    ) -> None:
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
            key = (match.local_keyframe_id, match.remote_frame)
            if key not in self.pending_matches:
                self.pending_matches.add(key)
                self.candidate_queue.append(match)

    def receive_keyframe(
        self,
        match: BowMatch,
        payload: Optional[KeyframePayload],
        error: str = "",
    ) -> None:
        with self.distributed_lock:
            self.response_queue.append((match, payload, error))

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

        if self.lc_in_progress:
            return

        response = None
        request = None
        with self.distributed_lock:
            if self.response_queue:
                response = self.response_queue.popleft()
            elif self.candidate_queue:
                request = self.candidate_queue.popleft()

        if response is not None:
            match, payload, error = response
            try:
                if payload is not None and not error:
                    try:
                        self._verify_remote_match(match, payload)
                    except (FileNotFoundError, KeyError):
                        # A peer can respond while the query's neighboring
                        # images are crossing the stable-cache boundary. The
                        # candidate may recur; it must not terminate odometry.
                        pass
            finally:
                with self.distributed_lock:
                    self.pending_matches.discard(
                        (match.local_keyframe_id, match.remote_frame)
                    )
        elif request is not None:
            self.transport.request_keyframe(request)

    @torch.no_grad()
    def _verify_remote_match(
        self,
        match: BowMatch,
        remote: KeyframePayload,
    ) -> bool:
        local = self._local_payload(match.local_keyframe_id)
        max_depth = self.cfg.MULTI_ROBOT_MAX_DEPTH
        local_mask = (local.points[:, 2] > 0) & (local.points[:, 2] < max_depth)
        remote_mask = (remote.points[:, 2] > 0) & (remote.points[:, 2] < max_depth)
        if (
            np.count_nonzero(local_mask) < self.cfg.MULTI_ROBOT_MIN_INLIERS
            or np.count_nonzero(remote_mask) < self.cfg.MULTI_ROBOT_MIN_INLIERS
        ):
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
            return False
        local_indices, remote_indices = matches.mT
        local_points = local.points[local_mask][local_indices.cpu().numpy()]
        remote_points = remote.points[remote_mask][remote_indices.cpu().numpy()]
        result = self.verifier.verify(local_points, remote_points)
        if not result.success:
            return False

        constraint = InterRobotConstraint(
            query_frame=local.frame,
            match_frame=remote.frame,
            query_pose=local.pose,
            match_pose=remote.pose,
            bow_score=match.score,
            rotation=result.rotation,
            translation=result.translation,
            quaternion_xyzw=Rotation.from_matrix(result.rotation).as_quat(),
            scale=result.scale,
            inliers=result.inliers,
            inlier_ratio=result.inlier_ratio,
            verification_method=result.method,
        )
        self.transport.publish_constraint(constraint)
        with self.distributed_lock:
            self.candidate_detector.confirm(match)
        self.inter_robot_lc_count += 1
        return True

    def terminate(self, n):
        super().terminate(n)
        print(f"INTER-ROBOT LC COUNT: {self.inter_robot_lc_count}")

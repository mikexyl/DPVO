"""Offline Sim(3) graph optimization using existing LieTorch and SciPy.

S_i maps anchor-camera RDF points to world: p_w = s_i R_i p_i + t_i.
An edge (i,j) measures Z_ji: p_j = Z_ji p_i. Its residual is
Log((S_j^-1 S_i) Z_ji^-1), in the target camera frame. No projection,
heading constraint, or saved-world-pose conjugation is used for sphere loops.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import torch

from .lietorch import Sim3


def as_sim3(values):
    return Sim3(torch.as_tensor(np.array(values, dtype=np.float64, copy=True)))


def validate_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 8 or not np.isfinite(poses).all():
        raise ValueError('expected finite Nx8 Sim3 values: t, quaternion xyzw, scale')
    if (poses[:, 7] <= 0).any() or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1., atol=1e-6):
        raise ValueError('positive scales and unit quaternions required')
    return poses.copy()


def from_tum(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError('expected finite Nx8 TUM rows')
    if len(rows) < 2 or (np.diff(rows[:, 0]) <= 0).any():
        raise ValueError('need at least two strictly ordered timestamps')
    poses = np.c_[rows[:, 1:], np.ones(len(rows))]
    # Saved quaternions have float32 export rounding; reject larger corruption.
    if not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1., atol=1e-5):
        raise ValueError('invalid saved quaternion')
    poses[:, 3:7] /= np.linalg.norm(poses[:, 3:7], axis=1)[:, None]
    return validate_poses(poses)


def measurement(rotation, translation, scale):
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('invalid Sim3 measurement')
    return np.r_[translation, Rotation.from_matrix(rotation).as_quat(), scale]


@dataclass
class Graph:
    initial: np.ndarray
    source: np.ndarray
    target: np.ndarray
    measurements: np.ndarray
    sigmas: np.ndarray
    loops: np.ndarray
    confidence: np.ndarray

    def __post_init__(self):
        self.initial = validate_poses(self.initial)
        self.measurements = validate_poses(self.measurements)
        if len(self.initial) < 2:
            raise ValueError('graph requires at least two poses')
        self.source, self.target = np.asarray(self.source), np.asarray(self.target)
        m = len(self.measurements)
        for index in (self.source, self.target):
            if (index.shape != (m,) or index.dtype.kind not in 'iu' or (index < 0).any()
                    or (index >= len(self.initial)).any()):
                raise ValueError('invalid edge endpoint')
        if (self.source == self.target).any():
            raise ValueError('self edges are not constraints')
        self.sigmas = np.asarray(self.sigmas, dtype=np.float64)
        self.loops = np.asarray(self.loops, dtype=bool)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        if (self.sigmas.shape != (m, 7) or not np.isfinite(self.sigmas).all()
                or (self.sigmas <= 0).any() or self.loops.shape != (m,)
                or self.confidence.shape != (m,) or not np.isfinite(self.confidence).all()
                or (self.confidence <= 0).any() or (self.confidence > 1).any()):
            raise ValueError('invalid graph weights')
        # One fixed node removes seven gauge freedoms only for a connected graph.
        neighbors = [set() for _ in self.initial]
        for i, j in zip(self.source, self.target):
            neighbors[i].add(j); neighbors[j].add(i)
        reached, pending = {0}, [0]
        while pending:
            for node in neighbors[pending.pop()] - reached:
                reached.add(node); pending.append(node)
        if len(reached) != len(self.initial):
            raise ValueError('graph must be connected to the fixed first node')


def edge_errors(graph, poses):
    states = as_sim3(poses)
    return ((states[graph.target].inv() * states[graph.source])
            * as_sim3(graph.measurements).inv()).log().numpy().copy()


def robust_residual(errors, graph, delta):
    """Quadratic odometry and an isotropic pseudo-Huber loss per 7D loop.

    Multiplying an entire block preserves rotational symmetry of the loss.
    Confidence multiplies the loss outside robustification, so it does not
    hide large errors by changing the robust transition threshold.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('robust delta must be positive')
    r = errors / graph.sigmas
    norm = np.linalg.norm(r[graph.loops], axis=1)
    factor = np.sqrt(2. / (np.hypot(1., norm / delta) + 1.))
    r[graph.loops] *= factor[:, None]
    return r * np.sqrt(graph.confidence[:, None])


def edge_statistics(graph, poses, delta):
    errors = edge_errors(graph, poses)
    norm = np.linalg.norm(errors / graph.sigmas, axis=1)
    weight = np.ones(len(norm))
    weight[graph.loops] = 1. / np.hypot(1., norm[graph.loops] / delta)
    return dict(log_translation_norm=np.linalg.norm(errors[:, :3], axis=1),
                rotation_degrees=np.degrees(np.linalg.norm(errors[:, 3:6], axis=1)),
                absolute_log_scale=np.abs(errors[:, 6]), normalized_norm=norm,
                robust_weight=weight, effective_weight=weight * graph.confidence)


def optimize_graph(graph, *, delta=3., max_nfev=400, workers=4, verbose=False):
    """Gauge-fixed sparse trust-region solve, with right Sim3 increments.

    LieTorch supplies the C++ Exp/Log/group operations already used by DPVO.
    SciPy uses sparse finite-difference Jacobians and LSMR. Input poses and
    global Torch thread settings are preserved. Returned arrays own storage.
    """
    if not 1 <= workers <= 32 or max_nfev < 1:
        raise ValueError('invalid solver budget')
    n, m = len(graph.initial), len(graph.measurements)
    base = as_sim3(graph.initial)
    inverse_measurements = as_sim3(graph.measurements).inv()
    sparsity = lil_matrix((m * 7, (n - 1) * 7), dtype=np.int8)
    for e, (i, j) in enumerate(zip(graph.source, graph.target)):
        for node in (i, j):
            if node:
                sparsity[e*7:(e+1)*7, (node-1)*7:node*7] = 1
    zero = torch.zeros((1, 7), dtype=torch.float64)

    def states(x):
        increments = torch.cat((zero, torch.from_numpy(x.reshape(-1, 7))), dim=0)
        return base * Sim3.exp(increments)

    start, last_print, calls, best = perf_counter(), 0., 0, float('inf')
    history = []

    def residual(x):
        nonlocal last_print, calls, best
        poses = states(x)
        errors = ((poses[graph.target].inv() * poses[graph.source]) * inverse_measurements).log().numpy()
        r = robust_residual(errors, graph, delta).ravel()
        cost = float(.5 * (r @ r))
        calls += 1
        if cost < best * (1 - 1e-7):
            best = cost
            history.append(dict(call=calls, seconds=perf_counter()-start, cost=cost))
        if verbose and perf_counter() - last_print > 15:
            print(f'PGO {perf_counter()-start:.1f}s / {calls} residual calls / best cost {best:.6g}', flush=True)
            last_print = perf_counter()
        return r

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(workers)
        x0 = np.zeros((n-1) * 7, dtype=np.float64)
        initial_residual = residual(x0)
        result = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(),
                               method='trf', tr_solver='lsmr', x_scale='jac',
                               ftol=1e-8, xtol=1e-8, gtol=1e-7,
                               tr_options=dict(atol=1e-10, btol=1e-10, maxiter=3000),
                               max_nfev=max_nfev)
        optimized = validate_poses(states(result.x).data.numpy())
    finally:
        torch.set_num_threads(previous_threads)
    np.testing.assert_array_equal(optimized[0], graph.initial[0])
    initial_cost = float(.5 * (initial_residual @ initial_residual))
    if not np.isfinite(result.cost) or result.cost > initial_cost + 1e-8:
        raise RuntimeError('optimization produced a nonfinite or increasing objective')
    summary = dict(success=bool(result.success), status=int(result.status), message=result.message,
                   initial_cost=initial_cost, final_cost=float(result.cost),
                   optimality=float(result.optimality), nfev=result.nfev, njev=result.njev,
                   residual_calls=calls, seconds=perf_counter()-start, history=history,
                   gauge='first camera-to-world Sim3 fixed exactly, including scale')
    return optimized, summary

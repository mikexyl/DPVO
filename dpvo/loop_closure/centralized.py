"""Centralized Sim(3) pose-graph optimization for robot map frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


RobotKey = tuple[str, str]


@dataclass
class Sim3:
    translation: np.ndarray
    rotation: np.ndarray
    scale: float = 1.0

    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.scale = float(self.scale)

    @classmethod
    def identity(cls):
        return cls(np.zeros(3), np.eye(3), 1.0)

    @classmethod
    def from_pose(cls, pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        return cls(pose[:3], Rotation.from_quat(pose[3:]).as_matrix(), 1.0)

    def compose(self, other: "Sim3") -> "Sim3":
        return Sim3(
            self.scale * (self.rotation @ other.translation) + self.translation,
            self.rotation @ other.rotation,
            self.scale * other.scale,
        )

    def inverse(self) -> "Sim3":
        rotation = self.rotation.T
        scale = 1.0 / self.scale
        return Sim3(
            -scale * (rotation @ self.translation),
            rotation,
            scale,
        )

    def apply(self, points):
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    def to_vector(self):
        return np.concatenate(
            [
                self.translation,
                Rotation.from_matrix(self.rotation).as_rotvec(),
                [np.log(self.scale)],
            ]
        )

    @classmethod
    def from_vector(cls, vector):
        vector = np.asarray(vector, dtype=np.float64).reshape(7)
        return cls(
            vector[:3],
            Rotation.from_rotvec(vector[3:6]).as_matrix(),
            np.exp(vector[6]),
        )


@dataclass
class RobotMapConstraint:
    query_robot: RobotKey
    match_robot: RobotKey
    query_pose: Sim3
    match_pose: Sim3
    query_to_match: Sim3
    weight: float = 1.0

    def map_measurement(self):
        """Return the measured query-map to match-map Sim(3)."""

        return (
            self.match_pose
            .compose(self.query_to_match)
            .compose(self.query_pose.inverse())
        )


@dataclass
class CentralizedPgoResult:
    transforms: dict[RobotKey, Sim3]
    success: bool
    cost: float
    residual_norm: float


class CentralizedRobotMapPGO:
    """Optimize global transforms for locally consistent robot maps."""

    def __init__(self, anchor: Optional[RobotKey] = None):
        self.anchor = anchor

    @staticmethod
    def _components(robots, constraints):
        adjacency = {robot: set() for robot in robots}
        for constraint in constraints:
            adjacency[constraint.query_robot].add(constraint.match_robot)
            adjacency[constraint.match_robot].add(constraint.query_robot)

        components = []
        unseen = set(robots)
        while unseen:
            root = min(unseen)
            stack = [root]
            component = set()
            while stack:
                robot = stack.pop()
                if robot in component:
                    continue
                component.add(robot)
                stack.extend(adjacency[robot] - component)
            unseen -= component
            components.append(component)
        return components

    @staticmethod
    def _initial_estimate(robots, constraints, anchors):
        estimates = {anchor: Sim3.identity() for anchor in anchors}
        changed = True
        while changed:
            changed = False
            for constraint in constraints:
                query = constraint.query_robot
                match = constraint.match_robot
                measurement = constraint.map_measurement()
                if match in estimates and query not in estimates:
                    estimates[query] = estimates[match].compose(measurement)
                    changed = True
                elif query in estimates and match not in estimates:
                    estimates[match] = estimates[query].compose(measurement.inverse())
                    changed = True
        for robot in robots:
            estimates.setdefault(robot, Sim3.identity())
        return estimates

    def solve(
        self,
        constraints: Iterable[RobotMapConstraint],
    ) -> CentralizedPgoResult:
        constraints = list(constraints)
        robots = sorted(
            {
                robot
                for constraint in constraints
                for robot in (constraint.query_robot, constraint.match_robot)
            }
        )
        if not robots:
            return CentralizedPgoResult({}, True, 0.0, 0.0)

        components = self._components(robots, constraints)
        anchors = {min(component) for component in components}
        if self.anchor in robots:
            component = next(x for x in components if self.anchor in x)
            anchors.discard(min(component))
            anchors.add(self.anchor)

        estimates = self._initial_estimate(robots, constraints, anchors)
        variables = [robot for robot in robots if robot not in anchors]
        if not variables:
            return CentralizedPgoResult(estimates, True, 0.0, 0.0)

        offsets = {robot: 7 * index for index, robot in enumerate(variables)}
        initial = np.concatenate([estimates[robot].to_vector() for robot in variables])

        def unpack(vector):
            transforms = {anchor: Sim3.identity() for anchor in anchors}
            for robot in variables:
                offset = offsets[robot]
                transforms[robot] = Sim3.from_vector(vector[offset : offset + 7])
            return transforms

        def residual(vector):
            transforms = unpack(vector)
            output = []
            for constraint in constraints:
                query = transforms[constraint.query_robot]
                match = transforms[constraint.match_robot]
                predicted = match.inverse().compose(query)
                error = constraint.map_measurement().inverse().compose(predicted)
                weight = np.sqrt(max(constraint.weight, 1e-3))
                output.extend(weight * error.translation)
                output.extend(weight * Rotation.from_matrix(error.rotation).as_rotvec())
                output.append(weight * np.log(error.scale))
            return np.asarray(output, dtype=np.float64)

        result = least_squares(
            residual,
            initial,
            loss="huber",
            f_scale=1.0,
            max_nfev=200,
        )
        final_residual = residual(result.x)
        return CentralizedPgoResult(
            unpack(result.x),
            bool(result.success),
            float(result.cost),
            float(np.linalg.norm(final_residual)),
        )

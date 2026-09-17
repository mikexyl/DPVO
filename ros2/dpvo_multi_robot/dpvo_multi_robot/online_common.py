"""ROS-independent lifecycle and session helpers for the live fleet."""
from __future__ import annotations

import os
import hashlib
import json
import re
import signal
import subprocess


_ID = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_-]*$')


def session_frame(robot: str, session: str) -> str:
    if not _ID.fullmatch(robot) or not _ID.fullmatch(session):
        raise ValueError('Robot and session IDs must be nonempty path-safe identifiers')
    return f'{robot}/{session}/map'


def parse_session_frame(frame: str) -> tuple[str, str]:
    parts = frame.split('/')
    if len(parts) != 3 or parts[2] != 'map':
        raise ValueError(f'Not a live session frame: {frame!r}')
    session_frame(parts[0], parts[1])
    return parts[0], parts[1]


def connected_robots(robots, pairs) -> bool:
    robots = set(robots)
    if not robots:
        return False
    neighbors = {robot: set() for robot in robots}
    for a, b in pairs:
        if a in robots and b in robots:
            neighbors[a].add(b)
            neighbors[b].add(a)
    reached, pending = set(), [min(robots)]
    while pending:
        robot = pending.pop()
        if robot not in reached:
            reached.add(robot)
            pending.extend(neighbors[robot] - reached)
    return reached == robots


def robot_components(robots, pairs):
    """Deterministic connected components, including isolated robots."""
    remaining = set(robots)
    neighbors = {r: set() for r in remaining}
    for a, b in pairs:
        if a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)
    components = []
    while remaining:
        reached, pending = set(), [min(remaining)]
        while pending:
            robot = pending.pop()
            if robot not in reached:
                reached.add(robot)
                pending.extend(neighbors[robot] - reached)
        remaining -= reached
        components.append(sorted(reached))
    return components


class WorkerProcess:
    """Own exactly one process group; Stop reaps its camera/CUDA workers."""

    def __init__(self, command: list[str], stop_timeout: float = 8.0):
        if not command:
            raise ValueError('Worker command is empty')
        self.command = list(command)
        self.stop_timeout = stop_timeout
        self.process = None
        self.last_exit = None

    @property
    def running(self) -> bool:
        if self.process is None:
            return False
        code = self.process.poll()
        if code is not None:
            self.last_exit = code
            # A failed parent can leave descendants holding the camera.
            self.stop()
            return False
        return True

    def start(self) -> bool:
        if self.running:
            return False
        self.last_exit = None
        self.process = subprocess.Popen(self.command, start_new_session=True)
        return True

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired:
            pass
        # Kill any descendants even if the group leader has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        self.process = None


def fleet_frame(sessions):
    """Identify the entire alignment snapshot, including the anchor session."""
    encoded = json.dumps(sessions, sort_keys=True, separators=(",", ":")).encode()
    return "world/fleet_" + hashlib.sha256(encoded).hexdigest()[:24]

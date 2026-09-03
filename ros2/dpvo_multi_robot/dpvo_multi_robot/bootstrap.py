"""Re-enter the DPVO Pixi environment from a system-Python ROS executable."""

import os
from pathlib import Path
import shlex
import shutil
import sys


def _dpvo_root():
    configured = os.environ.get("DPVO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pixi.toml").is_file():
        return source_root

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pixi.toml").is_file() and (candidate / "dpvo").is_dir():
            return candidate
    raise RuntimeError("Cannot find DPVO root; set the DPVO_ROOT environment variable")


def _exec_module(module):
    pixi = shutil.which("pixi")
    if pixi is None:
        raise RuntimeError("pixi is required to run the DPVO ROS 2 node")
    root = _dpvo_root()
    manifest = Path(
        os.environ.get("DPVO_PIXI_MANIFEST", root / "pixi.toml")
    ).expanduser().resolve()
    command = (
        f'eval "$({shlex.quote(pixi)} shell-hook --manifest-path '
        f'{shlex.quote(str(manifest))})"\n'
        f'exec python -m {shlex.quote(module)} "$@"'
    )
    os.execv(
        "/bin/bash",
        ["bash", "-c", command, "dpvo-pixi-bootstrap", *sys.argv[1:]],
    )


def main():
    _exec_module("dpvo_multi_robot.node")


def centralized_pgo_main():
    _exec_module("dpvo_multi_robot.centralized_pgo")


def cbs_pgo_main():
    _exec_module("dpvo_multi_robot.cbs_pgo")


def euroc_player_main():
    _exec_module("dpvo_multi_robot.euroc_player")


def tum_player_main():
    _exec_module("dpvo_multi_robot.tum_player")


def kitti_player_main():
    _exec_module("dpvo_multi_robot.kitti_player")


def newer_college_player_main():
    _exec_module("dpvo_multi_robot.newer_college_player")


def cu_multi_player_main():
    _exec_module("dpvo_multi_robot.cu_multi_player")


def s3e_player_main():
    _exec_module("dpvo_multi_robot.s3e_player")

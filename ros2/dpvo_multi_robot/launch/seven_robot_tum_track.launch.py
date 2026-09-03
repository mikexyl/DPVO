"""Stage-one seven-robot DPVO tracking with no inter-robot verification."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_implementation_path = Path(__file__).with_name(
    "four_robot_tum_track.launch.py"
)
_spec = spec_from_file_location(
    "dpvo_multi_robot_four_robot_tum_track", _implementation_path
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load launch implementation: {_implementation_path}")
_implementation = module_from_spec(_spec)
_spec.loader.exec_module(_implementation)
_implementation.ROBOTS = (
    "robot0",
    "robot1",
    "robot2",
    "robot3",
    "robot4",
    "robot5",
    "robot6",
)


def generate_launch_description():
    return _implementation.generate_launch_description()

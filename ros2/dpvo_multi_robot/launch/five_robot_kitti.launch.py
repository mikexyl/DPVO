"""Five-robot KITTI 00 launch with 200-frame adjacent overlaps."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BASE_PATH = Path(__file__).with_name("three_robot_kitti.launch.py")
_SPEC = spec_from_file_location("dpvo_three_robot_kitti_launch", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load shared KITTI launch definition: {_BASE_PATH}")
_BASE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

_BASE.ROBOTS = (
    ("robot0", "start0", "end0", "0", "1008"),
    ("robot1", "start1", "end1", "808", "1916"),
    ("robot2", "start2", "end2", "1716", "2825"),
    ("robot3", "start3", "end3", "2625", "3733"),
    ("robot4", "start4", "end4", "3533", "4541"),
)


def generate_launch_description():
    return _BASE.generate_launch_description()

"""Ten-robot KITTI 00 launch with 50-frame adjacent overlaps."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BASE_PATH = Path(__file__).with_name("three_robot_kitti.launch.py")
_SPEC = spec_from_file_location("dpvo_three_robot_kitti_launch", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load shared KITTI launch definition: {_BASE_PATH}")
_BASE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

_BASE.ROBOTS = (
    ("robot0", "start0", "end0", "0", "479"),
    ("robot1", "start1", "end1", "429", "933"),
    ("robot2", "start2", "end2", "883", "1387"),
    ("robot3", "start3", "end3", "1337", "1841"),
    ("robot4", "start4", "end4", "1791", "2295"),
    ("robot5", "start5", "end5", "2245", "2750"),
    ("robot6", "start6", "end6", "2700", "3204"),
    ("robot7", "start7", "end7", "3154", "3658"),
    ("robot8", "start8", "end8", "3608", "4112"),
    ("robot9", "start9", "end9", "4062", "4541"),
)


def generate_launch_description():
    return _BASE.generate_launch_description()

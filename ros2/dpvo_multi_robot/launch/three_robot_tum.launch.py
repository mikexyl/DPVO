"""Three-robot TUM RGB-D launch using the shared TUM launch implementation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_shared_path = Path(__file__).with_name("two_robot_tum.launch.py")
_shared_spec = spec_from_file_location("dpvo_two_robot_tum_launch", _shared_path)
if _shared_spec is None or _shared_spec.loader is None:
    raise ImportError(f"Cannot load shared TUM launch module: {_shared_path}")
_shared = module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)

_shared.ROBOTS = ("robot0", "robot1", "robot2")
_shared.DEFAULT_SEQUENCES = (
    "rgbd_dataset_freiburg1_360",
    "rgbd_dataset_freiburg1_floor",
    "rgbd_dataset_freiburg1_room",
)

generate_launch_description = _shared.generate_launch_description

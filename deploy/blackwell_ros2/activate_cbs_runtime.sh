#!/usr/bin/env bash

# Make the externally built CBS dependencies available to every command run
# through the Blackwell/ROS 2 Pixi environment. Keep the prefix overridable for
# workstations whose ROS 2 dependency workspace lives elsewhere.
cbs_dependency_prefix="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"
export DPVO_CBS_DEPENDENCY_PREFIX="$cbs_dependency_prefix"

cbs_prepend_library_path() {
  local directory="$1"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$directory:"*) ;;
    *) LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
}

cbs_prepend_library_path "$cbs_dependency_prefix/aria_viz/lib"
cbs_prepend_library_path "$cbs_dependency_prefix/aria_common/lib"
cbs_prepend_library_path "$cbs_dependency_prefix/gtsam/lib"
export LD_LIBRARY_PATH

unset cbs_dependency_prefix
unset -f cbs_prepend_library_path

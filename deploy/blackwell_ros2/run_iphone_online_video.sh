#!/usr/bin/env bash
# Use the existing built ROS overlay and Pixi environment with an optional
# isolated source tree. No dependency installation or build is performed.
set -eo pipefail
video_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
video_runtime_root="${DPVO_RUNTIME_ROOT:-$video_source_root}"
source "/opt/ros/${DPVO_ROS_DISTRO:-jazzy}/setup.bash"
source "${DPVO_ROS_INSTALL:-$video_runtime_root/build/ros2-install}/setup.bash"
export CONDA_PREFIX="$video_runtime_root/deploy/blackwell_ros2/.pixi/envs/default"
source "$video_runtime_root/deploy/blackwell_ros2/activate_cbs_runtime.sh"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$video_source_root:$video_source_root/ros2/dpvo_multi_robot:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-79}"
video_models_root="${DPVO_LEARNED_MODELS:-/data3/mikexyl/models/dpvo_loop_frontend}"
export TORCH_HOME="${TORCH_HOME:-$video_models_root/torch}"
export HF_HOME="${HF_HOME:-$video_models_root/huggingface}"
exec "$CONDA_PREFIX/bin/python" "$video_source_root/deploy/blackwell_ros2/iphone_online_video.py" "$@"

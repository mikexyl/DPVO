#!/usr/bin/env bash
set -eo pipefail

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/home/mikexyl/workspaces/dpvo_ws/src/DPVO}"
DPVO_DATA_ROOT="${DPVO_EUROC_ROOT:-/data3/mikexyl/datasets/euroc}"
DPVO_VOCAB="${DPVO_ORB_VOCAB:-/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt}"
DPVO_RESULTS="${DPVO_RESULTS_ROOT:-/data3/mikexyl/results/dpvo_multi_robot}"
DPVO_RESULT_TAG="${DPVO_OUTPUT_TAG:-mh_01_02_03_full}"

source /opt/ros/jazzy/setup.bash
source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-48}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/dpvo-mps-pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/dpvo-mps-log}"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-33}"
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

mkdir -p "$DPVO_RESULTS"
cd "$DPVO_DEPLOY_ROOT"

exec ros2 launch dpvo_multi_robot three_robot_euroc.launch.py \
  bag_root:="$DPVO_DATA_ROOT" \
  network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
  config:="$DPVO_DEPLOY_ROOT/config/fast.yaml" \
  orb_vocab:="$DPVO_VOCAB" \
  calib:="$DPVO_DEPLOY_ROOT/calib/euroc.txt" \
  stride:="${DPVO_STRIDE:-2}" \
  max_frames:="${DPVO_MAX_FRAMES:-0}" \
  bow_threshold:="${DPVO_BOW_THRESHOLD:-0.03}" \
  bow_repetitions:="${DPVO_BOW_REPETITIONS:-2}" \
  bow_nms_radius:="${DPVO_BOW_NMS_RADIUS:-15}" \
  teaser_noise_bound:="${DPVO_TEASER_NOISE_BOUND:-0.10}" \
  min_inliers:="${DPVO_MIN_INLIERS:-25}" \
  min_inlier_ratio:="${DPVO_MIN_INLIER_RATIO:-0.15}" \
  rerun_connect:="${DPVO_RERUN_CONNECT:-}" \
  rerun_recording_id:="${DPVO_RERUN_RECORDING_ID:-$DPVO_RESULT_TAG}" \
  pgo_output:="$DPVO_RESULTS/${DPVO_RESULT_TAG}_centralized.json"

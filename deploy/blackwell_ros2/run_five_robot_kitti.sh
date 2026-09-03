#!/usr/bin/env bash
set -eo pipefail

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/data3/dpvo_cbs_ws/src/DPVO}"
DPVO_DATA_ROOT="${DPVO_KITTI_ROOT:-/data1/mikexyl/datasets/kitti_odometry/dataset}"
DPVO_VOCAB="${DPVO_ORB_VOCAB:-/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt}"
DPVO_RESULTS="${DPVO_RESULTS_ROOT:-/data3/mikexyl/results/dpvo_multi_robot}"
DPVO_RESULT_TAG="${DPVO_OUTPUT_TAG:-kitti00_five_robot_overlap200_full_20260901}"
DPVO_CBS_DEPENDENCY_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"

source /opt/ros/jazzy/setup.bash
source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-48}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/dpvo-kitti-five-robot-mps-pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/dpvo-kitti-five-robot-mps-log}"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-20}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export LD_LIBRARY_PATH="$DPVO_CBS_DEPENDENCY_PREFIX/gtsam/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_common/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$DPVO_RESULTS"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

MPS_STARTED=0
if [[ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]]; then
  nvidia-cuda-mps-control -d
  MPS_STARTED=1
fi

stop_private_mps() {
  if [[ "$MPS_STARTED" == "1" && -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null
  fi
}
trap stop_private_mps EXIT

cd "$DPVO_DEPLOY_ROOT"

rerun_args=()
if [[ -n "${DPVO_RERUN_SAVE_DIR:-}" ]]; then
  mkdir -p "$DPVO_RERUN_SAVE_DIR"
  for robot_index in 0 1 2 3 4; do
    rerun_args+=("rerun_save${robot_index}:=$DPVO_RERUN_SAVE_DIR/robot${robot_index}.rrd")
  done
  rerun_args+=("rerun_recording_id:=${DPVO_RERUN_RECORDING_ID:-$DPVO_RESULT_TAG}")
elif [[ -n "${DPVO_RERUN_CONNECT:-}" ]]; then
  rerun_args+=(
    "rerun_connect:=$DPVO_RERUN_CONNECT"
    "rerun_recording_id:=${DPVO_RERUN_RECORDING_ID:-$DPVO_RESULT_TAG}"
  )
fi

ros2 launch dpvo_multi_robot five_robot_kitti.launch.py \
  dataset_root:="$DPVO_DATA_ROOT" \
  sequence:="${DPVO_KITTI_SEQUENCE:-00}" \
  image_dir:="${DPVO_KITTI_IMAGE_DIR:-image_0}" \
  calibration_key:="${DPVO_KITTI_CALIBRATION_KEY:-P0}" \
  start0:="${DPVO_START0:-0}" end0:="${DPVO_END0:-1008}" \
  start1:="${DPVO_START1:-808}" end1:="${DPVO_END1:-1916}" \
  start2:="${DPVO_START2:-1716}" end2:="${DPVO_END2:-2825}" \
  start3:="${DPVO_START3:-2625}" end3:="${DPVO_END3:-3733}" \
  start4:="${DPVO_START4:-3533}" end4:="${DPVO_END4:-4541}" \
  network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
  config:="$DPVO_DEPLOY_ROOT/config/fast.yaml" \
  orb_vocab:="$DPVO_VOCAB" \
  stride:="${DPVO_STRIDE:-2}" \
  image_scale:="${DPVO_IMAGE_SCALE:-0.75}" \
  max_frames:="${DPVO_MAX_FRAMES:-0}" \
  random_seed:="${DPVO_RANDOM_SEED:--1}" \
  retrieval_backend:="${DPVO_RETRIEVAL_BACKEND:-dbow2}" \
  local_feature_backend:="${DPVO_LOCAL_FEATURE_BACKEND:-disk}" \
  bow_threshold:="${DPVO_BOW_THRESHOLD:-0.03}" \
  bow_repetitions:="${DPVO_BOW_REPETITIONS:-2}" \
  bow_nms_radius:="${DPVO_BOW_NMS_RADIUS:-15}" \
  bow_backfill:="${DPVO_BOW_BACKFILL:-true}" \
  reserve_inflight:="${DPVO_RESERVE_INFLIGHT:-true}" \
  teaser_noise_bound:="${DPVO_TEASER_NOISE_BOUND:-0.10}" \
  min_inliers:="${DPVO_MIN_INLIERS:-25}" \
  min_inlier_ratio:="${DPVO_MIN_INLIER_RATIO:-0.15}" \
  max_depth:="${DPVO_MAX_DEPTH:-20.0}" \
  keyframe_max_attempts:="${DPVO_KEYFRAME_MAX_ATTEMPTS:-5}" \
  keyframe_retry_delay:="${DPVO_KEYFRAME_RETRY_DELAY:-0.5}" \
  loop_diagnostics_dir:="${DPVO_LOOP_DIAGNOSTICS_DIR:-$DPVO_RESULTS/${DPVO_RESULT_TAG}_loop_diagnostics}" \
  loop_diagnostics_period:="${DPVO_LOOP_DIAGNOSTICS_PERIOD:-5.0}" \
  "${rerun_args[@]}" \
  pgo_output:="$DPVO_RESULTS/${DPVO_RESULT_TAG}_centralized.json" \
  pose_graph_output:="${DPVO_POSE_GRAPH_OUTPUT:-$DPVO_RESULTS/${DPVO_RESULT_TAG}_pose_graph}" \
  pose_graph_export_period:="${DPVO_POSE_GRAPH_EXPORT_PERIOD:-0.0}" \
  pose_graph_odometry_weight:="${DPVO_POSE_GRAPH_ODOMETRY_WEIGHT:-100.0}" \
  enable_cbs:="${DPVO_ENABLE_CBS:-true}" \
  cbs_executable:="${DPVO_CBS_EXECUTABLE:-$DPVO_DEPLOY_ROOT/build/cbs/examples/cbs_dpvo_sim3_offline}" \
  cbs_output_dir:="${DPVO_CBS_OUTPUT_DIR:-$DPVO_RESULTS/${DPVO_RESULT_TAG}_cbs}" \
  cbs_iterations:="${DPVO_CBS_ITERATIONS:-1000}" \
  cbs_stage_mode:="${DPVO_CBS_STAGE_MODE:-alternating}" \
  cbs_pose_warmup_iterations:="${DPVO_CBS_POSE_WARMUP_ITERATIONS:-0}" \
  cbs_pose_block_iterations:="${DPVO_CBS_POSE_BLOCK_ITERATIONS:-20}" \
  cbs_anchor_block_iterations:="${DPVO_CBS_ANCHOR_BLOCK_ITERATIONS:-20}" \
  cbs_target_hellinger:="${DPVO_CBS_TARGET_HELLINGER:-0.1}" \
  cbs_d_reset:="${DPVO_CBS_D_RESET:-0.6}" \
  cbs_settle_seconds:="${DPVO_CBS_SETTLE_SECONDS:-30.0}" \
  cbs_run_centralized_baseline:="${DPVO_CBS_RUN_CENTRALIZED_BASELINE:-true}" \
  cbs_run_explicit_anchor_centralized_baseline:="${DPVO_CBS_RUN_EXPLICIT_ANCHOR_CENTRALIZED_BASELINE:-true}"

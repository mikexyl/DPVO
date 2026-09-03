#!/usr/bin/env bash
set -eo pipefail

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/home/mikexyl/workspaces/dpvo_ws/src/DPVO}"
DPVO_DATA_ROOT="${DPVO_TUM_ROOT:-/data3/mikexyl/datasets/tum_rgbd}"
DPVO_VOCAB="${DPVO_ORB_VOCAB:-/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt}"
DPVO_RESULTS="${DPVO_RESULTS_ROOT:-/data3/mikexyl/results/dpvo_multi_robot}"
DPVO_LEARNED_MODELS="${DPVO_LEARNED_MODEL_ROOT:-/data3/mikexyl/models/dpvo_loop_frontend}"
DPVO_RESULT_TAG="${DPVO_OUTPUT_TAG:-tum_fr1_room_three_robot_full}"
DPVO_CBS_DEPENDENCY_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"

unset COLCON_CURRENT_PREFIX
source /opt/ros/jazzy/setup.bash
source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"
export TORCH_HOME="${TORCH_HOME:-$DPVO_LEARNED_MODELS/torch}"
export HF_HOME="${HF_HOME:-$DPVO_LEARNED_MODELS/huggingface}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-49}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/dpvo-tum-room-mps-pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/dpvo-tum-room-mps-log}"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-33}"
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export LD_LIBRARY_PATH="$DPVO_CBS_DEPENDENCY_PREFIX/gtsam/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_common/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$DPVO_RESULTS"
cd "$DPVO_DEPLOY_ROOT"

rerun_args=()
if [[ -n "${DPVO_RERUN_CONNECT:-}" ]]; then
  rerun_args+=(
    "rerun_connect:=$DPVO_RERUN_CONNECT"
    "rerun_recording_id:=${DPVO_RERUN_RECORDING_ID:-$DPVO_RESULT_TAG}"
  )
fi

exec ros2 launch dpvo_multi_robot three_robot_tum.launch.py \
  dataset_root:="$DPVO_DATA_ROOT" \
  sequence0:="${DPVO_TUM_SEQUENCE0:-rgbd_dataset_freiburg1_360}" \
  sequence1:="${DPVO_TUM_SEQUENCE1:-rgbd_dataset_freiburg1_floor}" \
  sequence2:="${DPVO_TUM_SEQUENCE2:-rgbd_dataset_freiburg1_room}" \
  network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
  config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
  orb_vocab:="$DPVO_VOCAB" \
  stride:="${DPVO_STRIDE:-1}" \
  max_frames:="${DPVO_MAX_FRAMES:-0}" \
  retrieval_backend:="${DPVO_RETRIEVAL_BACKEND:-megaloc}" \
  megaloc_repo:="${DPVO_MEGALOC_REPO:-$DPVO_LEARNED_MODELS/MegaLoc}" \
  megaloc_model_id:="${DPVO_MEGALOC_MODEL_ID:-gmberton/MegaLoc@5fe0dd697c}" \
  megaloc_threshold:="${DPVO_MEGALOC_THRESHOLD:-0.18}" \
  local_feature_backend:="${DPVO_LOCAL_FEATURE_BACKEND:-xfeat}" \
  xfeat_repo:="${DPVO_XFEAT_REPO:-$DPVO_LEARNED_MODELS/accelerated_features}" \
  xfeat_top_k:="${DPVO_XFEAT_TOP_K:-2048}" \
  xfeat_detection_threshold:="${DPVO_XFEAT_DETECTION_THRESHOLD:-0.05}" \
  lightglue_min_confidence:="${DPVO_LIGHTGLUE_MIN_CONFIDENCE:-0.10}" \
  bow_threshold:="${DPVO_BOW_THRESHOLD:-0.012}" \
  bow_repetitions:="${DPVO_BOW_REPETITIONS:-1}" \
  bow_nms_radius:="${DPVO_BOW_NMS_RADIUS:-10}" \
  teaser_noise_bound:="${DPVO_TEASER_NOISE_BOUND:-0.10}" \
  min_inliers:="${DPVO_MIN_INLIERS:-15}" \
  min_inlier_ratio:="${DPVO_MIN_INLIER_RATIO:-0.15}" \
  max_depth:="${DPVO_MAX_DEPTH:-0.0}" \
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
  cbs_iterations:="${DPVO_CBS_ITERATIONS:-200}" \
  cbs_stage_mode:="${DPVO_CBS_STAGE_MODE:-alternating}" \
  cbs_settle_seconds:="${DPVO_CBS_SETTLE_SECONDS:-30.0}" \
  cbs_run_centralized_baseline:="${DPVO_CBS_RUN_CENTRALIZED_BASELINE:-true}" \
  cbs_run_explicit_anchor_centralized_baseline:="${DPVO_CBS_RUN_EXPLICIT_ANCHOR_CENTRALIZED_BASELINE:-true}"

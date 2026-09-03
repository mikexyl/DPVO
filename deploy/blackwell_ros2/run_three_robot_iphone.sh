#!/usr/bin/env bash
set -eo pipefail

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/data3/dpvo_cbs_ws/src/DPVO}"
DPVO_DATA_ROOT="${DPVO_IPHONE_ROOT:-/data3/dpvo_cbs_ws/data/iphone_302_three_robot}"
DPVO_RESULTS="${DPVO_RESULTS_ROOT:-/data3/dpvo_cbs_ws/results}"
DPVO_RESULT_TAG="${DPVO_OUTPUT_TAG:-iphone_302_1_2_3_three_robot_classic_20260831}"
DPVO_LEARNED_MODELS="${DPVO_LEARNED_MODEL_ROOT:-/data3/mikexyl/models/dpvo_loop_frontend}"
DPVO_CBS_DEPENDENCY_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"

unset COLCON_CURRENT_PREFIX
source /opt/ros/jazzy/setup.bash
source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"
export TORCH_HOME="${TORCH_HOME:-$DPVO_LEARNED_MODELS/torch}"
export HF_HOME="${HF_HOME:-$DPVO_LEARNED_MODELS/huggingface}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-62}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$DPVO_RESULTS/$DPVO_RESULT_TAG/ros_log}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-33}"
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export LD_LIBRARY_PATH="$DPVO_CBS_DEPENDENCY_PREFIX/gtsam/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_common/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"

RUN_DIR="$DPVO_RESULTS/$DPVO_RESULT_TAG"
mkdir -p "$RUN_DIR/loop_diagnostics" "$RUN_DIR/cbs" "$ROS_LOG_DIR"
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
  sequence0:="${DPVO_IPHONE_SEQUENCE0:-302_1}" \
  sequence1:="${DPVO_IPHONE_SEQUENCE1:-302_2}" \
  sequence2:="${DPVO_IPHONE_SEQUENCE2:-302_3}" \
  network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
  config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
  orb_vocab:="$DPVO_DATA_ROOT/ORBvoc.txt" \
  stride:="${DPVO_STRIDE:-1}" \
  image_scale:="${DPVO_IMAGE_SCALE:-0.5}" \
  max_frames:="${DPVO_MAX_FRAMES:-0}" \
  random_seed:="${DPVO_RANDOM_SEED:-1234}" \
  retrieval_backend:="${DPVO_RETRIEVAL_BACKEND:-dbow2}" \
  local_feature_backend:="${DPVO_LOCAL_FEATURE_BACKEND:-disk}" \
  bow_threshold:="${DPVO_BOW_THRESHOLD:-0.01}" \
  bow_repetitions:="${DPVO_BOW_REPETITIONS:-1}" \
  bow_nms_radius:="${DPVO_BOW_NMS_RADIUS:-10}" \
  bow_backfill:="${DPVO_BOW_BACKFILL:-true}" \
  teaser_noise_bound:="${DPVO_TEASER_NOISE_BOUND:-0.10}" \
  min_inliers:="${DPVO_MIN_INLIERS:-15}" \
  min_inlier_ratio:="${DPVO_MIN_INLIER_RATIO:-0.15}" \
  max_depth:="${DPVO_MAX_DEPTH:-20.0}" \
  loop_diagnostics_dir:="$RUN_DIR/loop_diagnostics" \
  loop_diagnostics_period:="${DPVO_LOOP_DIAGNOSTICS_PERIOD:-5.0}" \
  camera_crop_x:=0 \
  camera_crop_y:=0 \
  camera_fx:=727.10 \
  camera_fy:=727.10 \
  camera_cx:=960.0 \
  camera_cy:=540.0 \
  camera_k1:=0.00044 \
  camera_k2:=0.0 \
  camera_p1:=0.0 \
  camera_p2:=0.0 \
  camera_k3:=0.0 \
  "${rerun_args[@]}" \
  pgo_output:="$RUN_DIR/centralized.json" \
  pose_graph_output:="$RUN_DIR/iphone_302_pose_graph" \
  pose_graph_export_period:="${DPVO_POSE_GRAPH_EXPORT_PERIOD:-10.0}" \
  pose_graph_odometry_weight:="${DPVO_POSE_GRAPH_ODOMETRY_WEIGHT:-100.0}" \
  enable_cbs:="${DPVO_ENABLE_CBS:-true}" \
  cbs_executable:="${DPVO_CBS_EXECUTABLE:-$DPVO_DEPLOY_ROOT/build/cbs/examples/cbs_dpvo_sim3_offline}" \
  cbs_output_dir:="$RUN_DIR/cbs" \
  cbs_iterations:="${DPVO_CBS_ITERATIONS:-1000}" \
  cbs_stage_mode:="${DPVO_CBS_STAGE_MODE:-alternating}" \
  cbs_target_hellinger:="${DPVO_CBS_TARGET_HELLINGER:-0.1}" \
  cbs_d_reset:="${DPVO_CBS_D_RESET:-0.6}" \
  cbs_settle_seconds:="${DPVO_CBS_SETTLE_SECONDS:-15.0}" \
  cbs_run_centralized_baseline:="${DPVO_CBS_RUN_CENTRALIZED_BASELINE:-true}" \
  cbs_run_explicit_anchor_centralized_baseline:="${DPVO_CBS_RUN_EXPLICIT_ANCHOR_CENTRALIZED_BASELINE:-true}"

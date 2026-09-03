#!/usr/bin/env bash
set -euo pipefail

# Reuse the validated 302_4 through 302_11 tracking artifacts, track 302_12 as
# robot8, then run the robot-count-independent verification and DPGO stages.

STAGE="${1:-all}"
case "$STAGE" in
  stage1|stage2|stage3|all) ;;
  *)
    echo "usage: $0 [stage1|stage2|stage3|all]" >&2
    exit 2
    ;;
esac

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/data3/dpvo_cbs_ws/src/DPVO}"
DPVO_DATA_ROOT="${DPVO_IPHONE_ROOT:-/data3/dpvo_cbs_ws/data/iphone_302_4_12_nine_robot}"
DPVO_RESULTS="${DPVO_RESULTS_ROOT:-/data3/dpvo_cbs_ws/results}"
DPVO_RESULT_TAG="${DPVO_OUTPUT_TAG:-iphone_302_4_5_6_7_8_9_10_11_12_nine_robot_staged_20260831}"
DPVO_REUSE_ARTIFACT_ROOT="${DPVO_REUSE_ARTIFACT_ROOT:-/data3/dpvo_cbs_ws/results/iphone_302_4_5_6_7_8_9_10_11_eight_robot_20260831/tracking}"
DPVO_CBS_DEPENDENCY_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"
DPVO_PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"

RUN_DIR="$DPVO_RESULTS/$DPVO_RESULT_TAG"
ARTIFACT_ROOT="$RUN_DIR/tracking"
ROS_LOG_DIR="${ROS_LOG_DIR:-$RUN_DIR/ros_log}"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_IPHONE_ROOT="$DPVO_DATA_ROOT"
export DPVO_RESULTS_ROOT="$DPVO_RESULTS"
export DPVO_OUTPUT_TAG="$DPVO_RESULT_TAG"
export DPVO_PIXI_MANIFEST
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-63}"
export ROS_LOG_DIR
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export LD_LIBRARY_PATH="$DPVO_CBS_DEPENDENCY_PREFIX/gtsam/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_common/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/dpvo-iphone-nine-robot-mps-pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/dpvo-iphone-nine-robot-mps-log}"

mkdir -p "$RUN_DIR" "$ROS_LOG_DIR" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

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

run_stage1() {
  echo "[stage1] Reusing robots0-7 and tracking 302_12 as robot8"
  mkdir -p "$ARTIFACT_ROOT"
  for robot_id in robot0 robot1 robot2 robot3 robot4 robot5 robot6 robot7; do
    source_artifact="$DPVO_REUSE_ARTIFACT_ROOT/$robot_id"
    destination_artifact="$ARTIFACT_ROOT/$robot_id"
    if [[ ! -f "$source_artifact/manifest.json" ]]; then
      echo "missing reusable tracking artifact: $source_artifact" >&2
      exit 1
    fi
    if [[ -e "$destination_artifact" ]]; then
      echo "refusing to overwrite tracking artifact: $destination_artifact" >&2
      exit 1
    fi
    cp -al "$source_artifact" "$destination_artifact"
  done

  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${DPVO_TRACKING_MPS_PERCENTAGE:-100}"
  unset COLCON_CURRENT_PREFIX
  set +u
  source /opt/ros/jazzy/setup.bash
  source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"
  set -u
  rerun_args=()
  if [[ -n "${DPVO_RERUN_CONNECT:-}" ]]; then
    rerun_args+=(
      "rerun_connect:=$DPVO_RERUN_CONNECT"
      "rerun_recording_id:=${DPVO_RERUN_RECORDING_ID:-$DPVO_RESULT_TAG-tracking}"
    )
  fi
  ros2 launch dpvo_multi_robot robot8_tum_track.launch.py \
    dataset_root:="$DPVO_DATA_ROOT" \
    sequence0:="${DPVO_IPHONE_SEQUENCE8:-302_12}" \
    network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
    config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
    orb_vocab:="$DPVO_DATA_ROOT/ORBvoc.txt" \
    tracking_artifact_output:="$ARTIFACT_ROOT" \
    stride:="${DPVO_STRIDE:-1}" \
    image_scale:="${DPVO_IMAGE_SCALE:-0.5}" \
    max_frames:="${DPVO_MAX_FRAMES:-0}" \
    random_seed:="${DPVO_RANDOM_SEED:-1234}" \
    local_feature_backend:="${DPVO_LOCAL_FEATURE_BACKEND:-disk}" \
    bow_threshold:="${DPVO_BOW_THRESHOLD:-0.01}" \
    camera_crop_x:=0 camera_crop_y:=0 \
    camera_fx:=727.10 camera_fy:=727.10 \
    camera_cx:=960.0 camera_cy:=540.0 \
    camera_k1:=0.00044 camera_k2:=0.0 \
    camera_p1:=0.0 camera_p2:=0.0 camera_k3:=0.0 \
    "${rerun_args[@]}" \
    2>&1 | tee "$RUN_DIR/stage1_tracking.log"

  pixi run --manifest-path "$DPVO_PIXI_MANIFEST" python -m \
    dpvo.loop_closure.offline_multirobot assemble \
    --artifact-root "$ARTIFACT_ROOT" \
    --output-base "$ARTIFACT_ROOT/unoptimized_tracking_graph" \
    --anchor-robot robot0 \
    --odometry-weight "${DPVO_POSE_GRAPH_ODOMETRY_WEIGHT:-100.0}"
}

run_shared_stage() {
  "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/run_five_robot_iphone_staged.sh" "$1"
}

case "$STAGE" in
  stage1) run_stage1 ;;
  stage2) run_shared_stage stage2 ;;
  stage3) run_shared_stage stage3 ;;
  all)
    run_stage1
    run_shared_stage stage2
    run_shared_stage stage3
    ;;
esac

echo "Completed $STAGE: $RUN_DIR"

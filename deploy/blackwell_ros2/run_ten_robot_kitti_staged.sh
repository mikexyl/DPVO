#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
case "$STAGE" in
  stage1|stage2|stage3|all) ;;
  *)
    echo "usage: $0 [stage1|stage2|stage3|all]" >&2
    exit 2
    ;;
esac

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/data3/dpvo_cbs_ws/src/DPVO}"
DPVO_DATA_ROOT="${DPVO_KITTI_ROOT:-/data1/mikexyl/datasets/kitti_odometry/dataset}"
DPVO_VOCAB="${DPVO_ORB_VOCAB:-/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt}"
DPVO_RESULTS="${DPVO_RESULTS_ROOT:-/data3/mikexyl/results/dpvo_multi_robot}"
DPVO_KITTI_SEQUENCE="${DPVO_KITTI_SEQUENCE:-00}"
DPVO_KITTI_IMAGE_DIR="${DPVO_KITTI_IMAGE_DIR:-image_0}"
DPVO_KITTI_CALIBRATION_KEY="${DPVO_KITTI_CALIBRATION_KEY:-P0}"
DPVO_OVERLAP_FRAMES="${DPVO_KITTI_OVERLAP_FRAMES:-200}"
DPVO_RESULT_TAG="${DPVO_OUTPUT_TAG:-kitti${DPVO_KITTI_SEQUENCE}_ten_robot_overlap${DPVO_OVERLAP_FRAMES}_staged_full_20260901}"
DPVO_CBS_DEPENDENCY_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"
DPVO_PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"

ROBOT_IDS=(robot0 robot1 robot2 robot3 robot4 robot5 robot6 robot7 robot8 robot9)
case "$DPVO_KITTI_SEQUENCE:$DPVO_OVERLAP_FRAMES" in
  00:10)
    STARTS=(0 449 903 1357 1811 2265 2720 3174 3628 4082)
    ENDS=(459 913 1367 1821 2275 2730 3184 3638 4092 4541)
    ;;
  00:200)
    STARTS=(0 354 808 1262 1716 2170 2625 3079 3533 3987)
    ENDS=(554 1008 1462 1916 2370 2825 3279 3733 4187 4541)
    ;;
  00:50)
    STARTS=(0 429 883 1337 1791 2245 2700 3154 3608 4062)
    ENDS=(479 933 1387 1841 2295 2750 3204 3658 4112 4541)
    ;;
  05:200)
    STARTS=(0 176 452 728 1004 1280 1557 1833 2109 2385)
    ENDS=(376 652 928 1204 1480 1757 2033 2309 2585 2761)
    ;;
  05:50)
    STARTS=(0 251 527 803 1079 1355 1632 1908 2184 2460)
    ENDS=(301 577 853 1129 1405 1682 1958 2234 2510 2761)
    ;;
  *)
    echo "unsupported KITTI sequence/overlap combination: $DPVO_KITTI_SEQUENCE/$DPVO_OVERLAP_FRAMES" >&2
    exit 2
    ;;
esac

RUN_DIR="$DPVO_RESULTS/$DPVO_RESULT_TAG"
ARTIFACT_ROOT="$RUN_DIR/tracking"
GV_DIR="$RUN_DIR/geometric_verification"
DPGO_DIR="$RUN_DIR/dpgo"
ROS_LOG_DIR="${ROS_LOG_DIR:-$RUN_DIR/ros_log}"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-48}"
export ROS_LOG_DIR
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export LD_LIBRARY_PATH="$DPVO_CBS_DEPENDENCY_PREFIX/gtsam/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_common/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/dpvo-kitti-ten-staged-mps-pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/dpvo-kitti-ten-staged-mps-log}"

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
  sequence_dir="$DPVO_DATA_ROOT/sequences/$DPVO_KITTI_SEQUENCE"
  image_dir="$sequence_dir/$DPVO_KITTI_IMAGE_DIR"
  if [[ ! -d "$image_dir" || ! -f "$sequence_dir/times.txt" ]]; then
    echo "KITTI input is incomplete: $image_dir and $sequence_dir/times.txt are required" >&2
    exit 2
  fi
  echo "[stage1] Sequential DPVO tracking for KITTI $DPVO_KITTI_SEQUENCE in ten windows with ${DPVO_OVERLAP_FRAMES}-frame overlap"
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
  unset COLCON_CURRENT_PREFIX
  set +u
  source /opt/ros/jazzy/setup.bash
  source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"
  set -u

  mkdir -p "$ARTIFACT_ROOT" "$RUN_DIR/rrd"
  : > "$RUN_DIR/stage1_tracking.log"
  for robot_index in "${!ROBOT_IDS[@]}"; do
    robot_id="${ROBOT_IDS[$robot_index]}"
    echo "[stage1] $robot_id [${STARTS[$robot_index]}, ${ENDS[$robot_index]})" \
      | tee -a "$RUN_DIR/stage1_tracking.log"
    ros2 launch dpvo_multi_robot single_robot_kitti_track.launch.py \
      dataset_root:="$DPVO_DATA_ROOT" \
      sequence:="$DPVO_KITTI_SEQUENCE" \
      image_dir:="$DPVO_KITTI_IMAGE_DIR" \
      calibration_key:="$DPVO_KITTI_CALIBRATION_KEY" \
      robot_id:="$robot_id" \
      start_frame:="${STARTS[$robot_index]}" \
      end_frame:="${ENDS[$robot_index]}" \
      network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
      config:="$DPVO_DEPLOY_ROOT/config/fast.yaml" \
      orb_vocab:="$DPVO_VOCAB" \
      tracking_artifact_output:="$ARTIFACT_ROOT" \
      stride:="${DPVO_STRIDE:-2}" \
      image_scale:="${DPVO_IMAGE_SCALE:-0.75}" \
      max_frames:="${DPVO_MAX_FRAMES:-0}" \
      random_seed:="${DPVO_RANDOM_SEED:-1234}" \
      local_feature_backend:="${DPVO_LOCAL_FEATURE_BACKEND:-disk}" \
      bow_threshold:="${DPVO_BOW_THRESHOLD:-0.03}" \
      rerun_save:="$RUN_DIR/rrd/$robot_id.rrd" \
      rerun_recording_id:="$DPVO_RESULT_TAG-tracking" \
      2>&1 | tee -a "$RUN_DIR/stage1_tracking.log"
  done

  pixi run --manifest-path "$DPVO_PIXI_MANIFEST" python -m \
    dpvo.loop_closure.offline_multirobot assemble \
    --artifact-root "$ARTIFACT_ROOT" \
    --output-base "$ARTIFACT_ROOT/unoptimized_tracking_graph" \
    --anchor-robot robot0 \
    --odometry-weight "${DPVO_POSE_GRAPH_ODOMETRY_WEIGHT:-100.0}"
}

run_stage2() {
  echo "[stage2] Sequential top-1 DBoW2 -> DIsK/LightGlue -> TEASER++"
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
  pixi run --manifest-path "$DPVO_PIXI_MANIFEST" python -m \
    dpvo.loop_closure.offline_multirobot verify \
    --artifact-root "$ARTIFACT_ROOT" \
    --orb-vocab "$DPVO_VOCAB" \
    --output-dir "$GV_DIR" \
    --anchor-robot robot0 \
    --local-feature-backend "${DPVO_LOCAL_FEATURE_BACKEND:-disk}" \
    --bow-threshold "${DPVO_BOW_THRESHOLD:-0.03}" \
    --bow-repetitions "${DPVO_BOW_REPETITIONS:-2}" \
    --bow-nms-radius "${DPVO_BOW_NMS_RADIUS:-15}" \
    --teaser-noise-bound "${DPVO_TEASER_NOISE_BOUND:-0.10}" \
    --min-inliers "${DPVO_MIN_INLIERS:-25}" \
    --min-inlier-ratio "${DPVO_MIN_INLIER_RATIO:-0.15}" \
    --max-depth "${DPVO_MAX_DEPTH:-20.0}" \
    --odometry-weight "${DPVO_POSE_GRAPH_ODOMETRY_WEIGHT:-100.0}" \
    --random-seed "${DPVO_RANDOM_SEED:-1234}" \
    2>&1 | tee "$RUN_DIR/stage2_geometric_verification.log"
}

run_stage3() {
  pose_block_iterations="${DPVO_CBS_POSE_BLOCK_ITERATIONS:-20}"
  anchor_block_iterations="${DPVO_CBS_ANCHOR_BLOCK_ITERATIONS:-20}"
  target_hellinger="${DPVO_CBS_TARGET_HELLINGER:-0.1}"
  # This option initializes only centralized baselines, never CBS.
  bootstrap_robot_anchors="${DPVO_CBS_BOOTSTRAP_ROBOT_ANCHORS:-true}"
  case "$bootstrap_robot_anchors" in
    true|1|yes) bootstrap_robot_anchor_flag=(--bootstrap-robot-anchors) ;;
    false|0|no) bootstrap_robot_anchor_flag=(--no-bootstrap-robot-anchors) ;;
    *)
      echo "invalid DPVO_CBS_BOOTSTRAP_ROBOT_ANCHORS=$bootstrap_robot_anchors (expected true or false)" >&2
      exit 2
      ;;
  esac
  echo "[stage3] CBS ${pose_block_iterations}-pose/${anchor_block_iterations}-anchor blocks with fixed target Hellinger ${target_hellinger}, no global initialization; centralized-only robot-anchor bootstrap ${bootstrap_robot_anchors}; both consume the same graph"
  pixi run --manifest-path "$DPVO_PIXI_MANIFEST" python -m \
    dpvo.loop_closure.offline_dpgo \
    --input-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --output-dir "$DPGO_DIR" \
    --cbs-executable "${DPVO_CBS_EXECUTABLE:-$DPVO_DEPLOY_ROOT/build/cbs/examples/cbs_dpvo_sim3_offline}" \
    --iterations "${DPVO_CBS_ITERATIONS:-1000}" \
    --stage-mode "${DPVO_CBS_STAGE_MODE:-alternating}" \
    --anchor-start-iteration "${DPVO_CBS_ANCHOR_START_ITERATION:-30}" \
    --anchor-stage-probability "${DPVO_CBS_ANCHOR_STAGE_PROBABILITY:-0.5}" \
    --pose-warmup-iterations "${DPVO_CBS_POSE_WARMUP_ITERATIONS:-0}" \
    --pose-block-iterations "$pose_block_iterations" \
    --anchor-block-iterations "$anchor_block_iterations" \
    --target-hellinger "$target_hellinger" \
    "${bootstrap_robot_anchor_flag[@]}" \
    --contract-alpha "${DPVO_CBS_CONTRACT_ALPHA:-0.95}" \
    --d-reset "${DPVO_CBS_D_RESET:-0.6}" \
    --odom-scale-sigma "${DPVO_CBS_ODOM_SCALE_SIGMA:--1.0}" \
    --inter-loop-scale-sigma "${DPVO_CBS_INTER_LOOP_SCALE_SIGMA:--1.0}" \
    --huber-k "${DPVO_CBS_HUBER_K:--1.0}" \
    --centralized-max-iterations "${DPVO_CENTRALIZED_MAX_ITERATIONS:-300}" \
    --random-seed "${DPVO_CBS_RANDOM_SEED:-42}" \
    2>&1 | tee "$RUN_DIR/stage3_dpgo.log"
}

case "$STAGE" in
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  stage3) run_stage3 ;;
  all)
    run_stage1
    run_stage2
    run_stage3
    ;;
esac

echo "Completed $STAGE: $RUN_DIR"

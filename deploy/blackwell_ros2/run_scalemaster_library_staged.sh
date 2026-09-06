#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <stage1|stage2|stage3|evaluate|plot|all>" >&2
  exit 2
fi
STAGE="$1"
case "$STAGE" in
  stage1|stage2|stage3|evaluate|plot|all) ;;
  *) echo "invalid stage: $STAGE" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPVO_DEPLOY_ROOT="${DPVO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$DPVO_DEPLOY_ROOT"
SOURCE_ROOT="${DPVO_SCALEMASTER_SOURCE_ROOT:-/data/scalemaster}"
RUN_DIR="${DPVO_SCALEMASTER_RESULT_ROOT:-$DPVO_DEPLOY_ROOT/results/scalemaster_library_three_staged_stride5_20260903}"
FRAME_STRIDE="${DPVO_SCALEMASTER_STRIDE:-5}"
IMAGE_SCALE="${DPVO_SCALEMASTER_IMAGE_SCALE:-0.5}"
ROBOT_IDS=(robot0 robot1 robot2)
OPTIMIZED_ROBOT_IDS=(robot0 robot1)
SEQUENCE_NAMES=(Library_01 Library_02 Library_03)
read -r -a EXPECTED_FRAMES <<< "${DPVO_SCALEMASTER_EXPECTED_FRAMES:-1303 1001 628}"
if [[ ${#EXPECTED_FRAMES[@]} -ne 3 ]]; then
  echo "DPVO_SCALEMASTER_EXPECTED_FRAMES must contain exactly three counts" >&2
  exit 2
fi

if [[ -n "${DPVO_ORB_VOCAB:-}" ]]; then
  DPVO_VOCAB="$DPVO_ORB_VOCAB"
elif [[ -f "$DPVO_DEPLOY_ROOT/ORBvoc.txt" ]]; then
  DPVO_VOCAB="$DPVO_DEPLOY_ROOT/ORBvoc.txt"
else
  DPVO_VOCAB="$DPVO_DEPLOY_ROOT/results/iphone_302_three_robot_dataset_20260830/ORBvoc.txt"
fi
PIXI_MANIFEST="${DPVO_PIXI_MANIFEST:-$DPVO_DEPLOY_ROOT/pixi.toml}"
PIXI_BIN="${DPVO_PIXI_BIN:-/home/mikexyl/.pixi/bin/pixi}"
ROS_DISTRO_NAME="${DPVO_ROS_DISTRO:-humble}"
ROS_INSTALL="${DPVO_ROS_INSTALL:-$DPVO_DEPLOY_ROOT/build/ros2-${ROS_DISTRO_NAME}-install}"
CBS_EXECUTABLE="${DPVO_CBS_EXECUTABLE:-$DPVO_DEPLOY_ROOT/build/cbs-explicit-anchor/examples/cbs_dpvo_sim3_offline}"
GTSAM_LIBRARY_DIR="${DPVO_GTSAM_LIBRARY_DIR:-}"
if [[ -z "$GTSAM_LIBRARY_DIR" ]]; then
  GTSAM_LIBRARY="$(ldd "$CBS_EXECUTABLE" 2>/dev/null | awk '$1 ~ /^libgtsam/ {print $3; exit}')"
  if [[ -n "$GTSAM_LIBRARY" && -f "$GTSAM_LIBRARY" ]]; then
    GTSAM_LIBRARY_DIR="$(dirname "$GTSAM_LIBRARY")"
  fi
fi
if [[ -z "$GTSAM_LIBRARY_DIR" || ! -d "$GTSAM_LIBRARY_DIR" ]]; then
  echo "set DPVO_GTSAM_LIBRARY_DIR to the directory containing the GTSAM runtime libraries" >&2
  exit 2
fi

ARTIFACT_ROOT="$RUN_DIR/tracking"
RRD_ROOT="$ARTIFACT_ROOT/rrd"
GV_DIR="$RUN_DIR/geometric_verification/three"
DPGO_DIR="$RUN_DIR/dpgo/three"
EVAL_DIR="$RUN_DIR/evaluation/three"
PLOTS_DIR="$RUN_DIR/plots/three"
STAGE1_GATE="$ARTIFACT_ROOT/stage1_gate_three.json"
GRAPH_GATE="$GV_DIR/graph_gate.json"
COMPONENT_GRAPH="$GV_DIR/unoptimized_verified_graph_robot0_robot1.json"
COMPONENT_PROVENANCE="$GV_DIR/unoptimized_verified_graph_robot0_robot1_provenance.json"
COMPONENT_GATE="$GV_DIR/component_graph_gate.json"
METRICS="$EVAL_DIR/metrics.json"
FINAL_GATE="$EVAL_DIR/final_gate.json"
ROS_LOG_DIR="${ROS_LOG_DIR:-$RUN_DIR/ros_log}"
GATE_HELPER="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/newer_college_staged.py"
EXPERIMENT_HELPER="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/scalemaster_staged.py"

export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST="$PIXI_MANIFEST"
export PATH="$(dirname "$PIXI_BIN"):$PATH"
export LD_LIBRARY_PATH="$GTSAM_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-79}"
export ROS_LOG_DIR
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"

mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT" "$RRD_ROOT" "$ROS_LOG_DIR"
PYTHON=("$PIXI_BIN" run --manifest-path "$PIXI_MANIFEST" python)
SEQUENCES=()
for sequence_name in "${SEQUENCE_NAMES[@]}"; do
  sequence="$SOURCE_ROOT/$sequence_name"
  SEQUENCES+=("$sequence")
done
for path in "$PIXI_BIN" "$PIXI_MANIFEST" "$DPVO_VOCAB" "$DPVO_DEPLOY_ROOT/dpvo.pth" \
  "$DPVO_DEPLOY_ROOT/config/default.yaml" "$GATE_HELPER" "$EXPERIMENT_HELPER" \
  "$CBS_EXECUTABLE" "${SEQUENCES[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done

setup_ros() {
  unset COLCON_CURRENT_PREFIX
  set +u
  source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
  source "$ROS_INSTALL/setup.bash"
  set -u
}

record_manifest() {
  local args=(record-manifest --run-dir "$RUN_DIR" --vocabulary "$DPVO_VOCAB"
    --network "$DPVO_DEPLOY_ROOT/dpvo.pth"
    --config "$DPVO_DEPLOY_ROOT/config/default.yaml")
  for index in "${!ROBOT_IDS[@]}"; do
    args+=(--source "${ROBOT_IDS[$index]}=${SEQUENCES[$index]}")
  done
  local parameters=(
    "tracking.stride=$FRAME_STRIDE"
    "tracking.image_scale=$IMAGE_SCALE"
    "tracking.random_seed=1234"
    "tracking.classic_intra_robot_loop_closure=true"
    "tracking.execution=sequential"
    "tracking.input_encoding=bgr8"
    "tracking.source_resolution=[1920,1440]"
    "stage2.retrieval=dbow2_top1"
    "stage2.bow_threshold=0.01"
    "stage2.bow_repetitions=1"
    "stage2.bow_nms_radius=10"
    "stage2.local_features=disk"
    "stage2.matcher=lightglue"
    "stage2.teaser_noise_bound=0.10"
    "stage2.min_inliers=15"
    "stage2.min_inlier_ratio=0.15"
    "stage2.max_depth=20.0"
    "stage2.odometry_weight=100.0"
    "stage2.random_seed=1234"
    'stage2.expected_components=[["robot0","robot1"],["robot2"]]'
    'stage2.negative_control=robot2'
    'stage3.optimized_component=["robot0","robot1"]'
    "runtime.gtsam_library_dir=$GTSAM_LIBRARY_DIR"
    "stage3.iterations=1000"
    "stage3.mode=alternating"
    "stage3.pose_block_iterations=20"
    "stage3.anchor_block_iterations=20"
    "stage3.target_hellinger=0.1"
    "stage3.d_reset=0.6"
    "stage3.random_seed=42"
    "stage3.huber_k=-1.0"
    "evaluation.scope=no_ground_truth"
    "evaluation.report_solver_agreement=true"
  )
  for parameter in "${parameters[@]}"; do args+=(--parameter "$parameter"); done
  "${PYTHON[@]}" "$EXPERIMENT_HELPER" "${args[@]}"
}

artifact_ready() {
  local index="$1"
  "${PYTHON[@]}" "$GATE_HELPER" artifact-ready \
    --tracking-root "$ARTIFACT_ROOT" --robot-id "${ROBOT_IDS[$index]}" \
    --expected-frames "${EXPECTED_FRAMES[$index]}" --tolerance 0 \
    --rrd "$RRD_ROOT/${ROBOT_IDS[$index]}.rrd"
}

run_preflight() {
  if [[ -f "$ARTIFACT_ROOT/robot0/manifest.json" ]]; then return; fi
  local root="$RUN_DIR/preflight/library01_stride${FRAME_STRIDE}_scale${IMAGE_SCALE}"
  if [[ -f "$root/robot0/manifest.json" ]]; then
    "${PYTHON[@]}" "$GATE_HELPER" artifact-ready --tracking-root "$root" \
      --robot-id robot0 --expected-frames 200 --tolerance 0
    return
  fi
  if [[ -d "$root/robot0" ]]; then
    echo "incomplete preflight exists; preserving it: $root/robot0" >&2
    exit 2
  fi
  ros2 launch dpvo_multi_robot single_robot_scalemaster_track.launch.py \
    sequence_dir:="${SEQUENCES[0]}" robot_id:=robot0 \
    network:="$DPVO_DEPLOY_ROOT/dpvo.pth" config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
    orb_vocab:="$DPVO_VOCAB" tracking_artifact_output:="$root" \
    stride:="$FRAME_STRIDE" start_frame:=100 max_frames:=200 \
    image_scale:="$IMAGE_SCALE" random_seed:=1234 local_feature_backend:=disk \
    bow_threshold:=0.01 2>&1 | tee "$RUN_DIR/preflight_library01.log"
  "${PYTHON[@]}" "$GATE_HELPER" artifact-ready --tracking-root "$root" \
    --robot-id robot0 --expected-frames 200 --tolerance 0
}

track_robot() {
  local index="$1"
  local robot_id="${ROBOT_IDS[$index]}"
  if [[ -f "$ARTIFACT_ROOT/$robot_id/manifest.json" ]]; then
    echo "[stage1] Reusing immutable $robot_id artifact"
    artifact_ready "$index"
    return
  fi
  if [[ -d "$ARTIFACT_ROOT/$robot_id" ]]; then
    echo "incomplete artifact exists; preserving it: $ARTIFACT_ROOT/$robot_id" >&2
    exit 2
  fi
  ros2 launch dpvo_multi_robot single_robot_scalemaster_track.launch.py \
    sequence_dir:="${SEQUENCES[$index]}" robot_id:="$robot_id" \
    network:="$DPVO_DEPLOY_ROOT/dpvo.pth" config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
    orb_vocab:="$DPVO_VOCAB" tracking_artifact_output:="$ARTIFACT_ROOT" \
    stride:="$FRAME_STRIDE" start_frame:=0 max_frames:=0 image_scale:="$IMAGE_SCALE" \
    random_seed:=1234 local_feature_backend:=disk bow_threshold:=0.01 \
    rerun_save:="$RRD_ROOT/$robot_id.rrd" \
    rerun_recording_id:="scalemaster-${SEQUENCE_NAMES[$index]}-$robot_id" \
    2>&1 | tee "$RUN_DIR/stage1_${robot_id}.log"
  artifact_ready "$index"
}

validate_stage1() {
  local args=(validate-stage1 --tracking-root "$ARTIFACT_ROOT" --rrd-root "$RRD_ROOT"
    --tracking-graph "$ARTIFACT_ROOT/unoptimized_tracking_graph_three.json"
    --robot-ids "${ROBOT_IDS[@]}" --tolerance 0 --output "$STAGE1_GATE")
  for index in "${!ROBOT_IDS[@]}"; do
    args+=(--expected "${ROBOT_IDS[$index]}=${EXPECTED_FRAMES[$index]}")
  done
  "${PYTHON[@]}" "$GATE_HELPER" "${args[@]}"
}

run_stage1() {
  echo "[stage1] ScaleMaster Library_01/02/03 at stride $FRAME_STRIDE"
  record_manifest
  setup_ros
  run_preflight
  for index in "${!ROBOT_IDS[@]}"; do track_robot "$index"; done
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_multirobot assemble \
    --artifact-root "$ARTIFACT_ROOT" --robot-ids "${ROBOT_IDS[@]}" \
    --output-base "$ARTIFACT_ROOT/unoptimized_tracking_graph_three" \
    --anchor-robot robot0 --odometry-weight 100.0
  validate_stage1
}

check_graph() {
  "${PYTHON[@]}" "$GATE_HELPER" check-graph \
    --graph "$GV_DIR/unoptimized_verified_graph.json" --robot-ids "${ROBOT_IDS[@]}" \
    --expected-component robot0,robot1 --expected-component robot2 \
    --min-pair robot0:robot1=1 --output "$GRAPH_GATE"
}

ensure_component_graph() {
  if [[ ! -s "$COMPONENT_GRAPH" || ! -s "$COMPONENT_PROVENANCE" ]]; then
    "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/subset_verified_graph.py" \
      --input "$GV_DIR/unoptimized_verified_graph.json" \
      --output-base "${COMPONENT_GRAPH%.json}" \
      --robot-ids "${OPTIMIZED_ROBOT_IDS[@]}" --anchor-robot robot0
  fi
  "${PYTHON[@]}" "$GATE_HELPER" check-graph --graph "$COMPONENT_GRAPH" \
    --robot-ids "${OPTIMIZED_ROBOT_IDS[@]}" --min-pair robot0:robot1=1 \
    --output "$COMPONENT_GATE"
}

run_stage2() {
  validate_stage1
  if [[ -s "$GV_DIR/verification.json" && -s "$GV_DIR/unoptimized_verified_graph.json" ]]; then
    check_graph
    ensure_component_graph
    return
  fi
  if [[ -d "$GV_DIR" ]] && find "$GV_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial stage-two directory exists; preserving it: $GV_DIR" >&2
    exit 2
  fi
  mkdir -p "$GV_DIR"
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_multirobot verify \
    --artifact-root "$ARTIFACT_ROOT" --robot-ids "${ROBOT_IDS[@]}" \
    --orb-vocab "$DPVO_VOCAB" --output-dir "$GV_DIR" --anchor-robot robot0 \
    --local-feature-backend disk --bow-threshold 0.01 --bow-repetitions 1 \
    --bow-nms-radius 10 --teaser-noise-bound 0.10 --min-inliers 15 \
    --min-inlier-ratio 0.15 --max-depth 20.0 --odometry-weight 100.0 \
    --random-seed 1234 2>&1 | tee "$RUN_DIR/stage2_three.log"
  check_graph
  ensure_component_graph
}

run_stage3() {
  validate_stage1
  check_graph
  ensure_component_graph
  if [[ -f "$DPGO_DIR/offline_dpgo_provenance.json" ]]; then
    if "${PYTHON[@]}" -c '
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if value.get("status") == "complete" and value.get("return_code") == 0 and value.get("copied_input_is_byte_identical") is True else 1)
' "$DPGO_DIR/offline_dpgo_provenance.json"; then
      return
    fi
    echo "incomplete stage-three provenance exists; preserving it: $DPGO_DIR" >&2
    exit 2
  fi
  if [[ -d "$DPGO_DIR" ]] && find "$DPGO_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial stage-three directory exists; preserving it: $DPGO_DIR" >&2
    exit 2
  fi
  mkdir -p "$DPGO_DIR"
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_dpgo \
    --input-graph "$COMPONENT_GRAPH" --output-dir "$DPGO_DIR" \
    --cbs-executable "$CBS_EXECUTABLE" --iterations 1000 --stage-mode alternating \
    --anchor-start-iteration 0 --anchor-stage-probability 0.5 \
    --pose-warmup-iterations 0 --pose-block-iterations 20 \
    --anchor-block-iterations 20 --target-hellinger 0.1 --contract-alpha 0.95 \
    --d-reset 0.6 --odom-scale-sigma -1.0 --inter-loop-scale-sigma -1.0 \
    --huber-k -1.0 --centralized-max-iterations 300 --random-seed 42 \
    --write-rerun-rrd 2>&1 | tee "$RUN_DIR/stage3_three.log"
}

final_gate() {
  local args=(final-gate --stage1-gate "$STAGE1_GATE" --graph-gate "$GRAPH_GATE"
    --verified-graph "$GV_DIR/unoptimized_verified_graph.json"
    --component-graph "$COMPONENT_GRAPH" --component-provenance "$COMPONENT_PROVENANCE"
    --component-graph-gate "$COMPONENT_GATE" --dpgo-dir "$DPGO_DIR"
    --metrics "$METRICS" --robot-ids "${ROBOT_IDS[@]}"
    --optimized-robot-ids "${OPTIMIZED_ROBOT_IDS[@]}"
    --output "$FINAL_GATE")
  if [[ "${1:-}" == plots ]]; then args+=(--plots-dir "$PLOTS_DIR"); fi
  "${PYTHON[@]}" "$EXPERIMENT_HELPER" "${args[@]}"
}

run_evaluate() {
  validate_stage1
  check_graph
  if [[ -f "$METRICS" ]]; then final_gate; return; fi
  if [[ -d "$EVAL_DIR" ]] && find "$EVAL_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial evaluation exists; preserving it: $EVAL_DIR" >&2
    exit 2
  fi
  mkdir -p "$EVAL_DIR"
  "${PYTHON[@]}" "$EXPERIMENT_HELPER" collect-metrics \
    --dpgo-dir "$DPGO_DIR" --robot-ids "${ROBOT_IDS[@]}" \
    --optimized-robot-ids "${OPTIMIZED_ROBOT_IDS[@]}" --output "$METRICS"
  final_gate
}

run_plot() {
  final_gate
  if [[ -f "$PLOTS_DIR/scalemaster_component_trajectories_loops.png" && -f "$PLOTS_DIR/cbs_sparse_joint_map.rrd" ]]; then
    final_gate plots
    return
  fi
  if [[ -d "$PLOTS_DIR" ]] && find "$PLOTS_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial plot directory exists; preserving it: $PLOTS_DIR" >&2
    exit 2
  fi
  mkdir -p "$PLOTS_DIR"
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/plot_newer_college.py" \
    --graph "$COMPONENT_GRAPH" --solver-dir "$DPGO_DIR" \
    --evo-dir "$EVAL_DIR" --output-dir "$PLOTS_DIR" --dataset-label ScaleMaster \
    --output-prefix scalemaster_component --alignment-mode solver-frame
  local npz_args=()
  for robot_id in "${OPTIMIZED_ROBOT_IDS[@]}"; do
    npz_args+=(--input-npz "$robot_id=$ARTIFACT_ROOT/$robot_id/map.npz")
  done
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/visualize_cbs_joint_map.py" \
    "${npz_args[@]}" --input-graph "$COMPONENT_GRAPH" \
    --cbs-csv "$DPGO_DIR/cbs.csv" \
    --comparison-csv "Explicit-anchor centralized=$DPGO_DIR/centralized_explicit_anchors.csv" \
    --output-rrd "$PLOTS_DIR/cbs_sparse_joint_map.rrd" \
    --output-ply "$PLOTS_DIR/cbs_sparse_joint_map.ply" \
    --output-manifest "$PLOTS_DIR/cbs_sparse_joint_map.json" \
    --recording-id scalemaster-three-cbs-sparse-map --host-distance-quantile 0.90 \
    --warp-mode posewise
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/visualize_cbs_joint_map.py" \
    "${npz_args[@]}" --input-graph "$COMPONENT_GRAPH" \
    --cbs-csv "$DPGO_DIR/centralized_explicit_anchors.csv" \
    --primary-label "Explicit-anchor centralized" --comparison-csv "CBS=$DPGO_DIR/cbs.csv" \
    --output-rrd "$PLOTS_DIR/centralized_sparse_joint_map.rrd" \
    --output-ply "$PLOTS_DIR/centralized_sparse_joint_map.ply" \
    --output-manifest "$PLOTS_DIR/centralized_sparse_joint_map.json" \
    --recording-id scalemaster-three-centralized-sparse-map \
    --host-distance-quantile 0.90 --warp-mode posewise
  final_gate plots
}

case "$STAGE" in
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  stage3) run_stage3 ;;
  evaluate) run_evaluate ;;
  plot) run_plot ;;
  all) run_stage1; run_stage2; run_stage3; run_evaluate; run_plot ;;
esac

echo "Completed ScaleMaster Library_01/02/03 $STAGE: $RUN_DIR"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <stage1|stage2|stage3|evaluate|plot|all> <two|three>" >&2
  exit 2
fi
STAGE="$1"
SCENARIO="$2"
case "$STAGE" in
  stage1|stage2|stage3|evaluate|plot|all) ;;
  *)
    echo "usage: $0 <stage1|stage2|stage3|evaluate|plot|all> <two|three>" >&2
    exit 2
    ;;
esac
case "$SCENARIO" in
  two) ROBOT_IDS=(robot0 robot1) ;;
  three) ROBOT_IDS=(robot0 robot1 robot2) ;;
  *)
    echo "usage: $0 <stage1|stage2|stage3|evaluate|plot|all> <two|three>" >&2
    exit 2
    ;;
esac

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/data3/dpvo_cbs_ws/src/DPVO}"
DPVO_DATA_ROOT="${DPVO_NCD_ROOT:-/data3/mikexyl/datasets/newer_college/collection1}"
RUN_DIR="${DPVO_NCD_RESULT_ROOT:-/data3/mikexyl/results/dpvo_multi_robot/newer_college_quad_staged_20260902}"
DPVO_VOCAB="${DPVO_ORB_VOCAB:-/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt}"
DPVO_PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"
DPVO_CBS_DEPENDENCY_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"
SOURCE_TOPIC="/alphasense_driver_ros/cam0/compressed"

ARTIFACT_ROOT="$RUN_DIR/tracking"
RRD_ROOT="$ARTIFACT_ROOT/rrd"
GV_DIR="$RUN_DIR/geometric_verification/$SCENARIO"
DPGO_DIR="$RUN_DIR/dpgo/$SCENARIO"
EVO_DIR="$RUN_DIR/evo/$SCENARIO"
PLOTS_DIR="$RUN_DIR/plots/$SCENARIO"
STAGE1_GATE="$ARTIFACT_ROOT/stage1_gate_${SCENARIO}.json"
GRAPH_GATE="$GV_DIR/graph_gate.json"
METRICS="$EVO_DIR/metrics.json"
FINAL_GATE="$EVO_DIR/expansion_gate.json"
ROS_LOG_DIR="${ROS_LOG_DIR:-$RUN_DIR/ros_log}"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
export ROS_LOG_DIR
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export LD_LIBRARY_PATH="$DPVO_CBS_DEPENDENCY_PREFIX/gtsam/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_common/lib:$DPVO_CBS_DEPENDENCY_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT" "$RRD_ROOT" "$ROS_LOG_DIR"

resolve_bag() {
  local label="$1"
  local explicit="$2"
  local token="$3"
  if [[ -n "$explicit" ]]; then
    if [[ ! -f "$explicit" ]]; then
      echo "$label does not exist: $explicit" >&2
      return 2
    fi
    realpath "$explicit"
    return
  fi
  local matches=()
  mapfile -d '' matches < <(
    find "$DPVO_DATA_ROOT" -maxdepth 6 -type f -iname "*quad*${token}*.bag" -print0
  )
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "expected one $label under $DPVO_DATA_ROOT, found ${#matches[@]}; set its DPVO_NCD_* environment variable" >&2
    printf '%s\n' "${matches[@]}" >&2
    return 2
  fi
  realpath "${matches[0]}"
}

resolve_groundtruth() {
  local label="$1"
  local explicit="$2"
  local token="$3"
  if [[ -n "$explicit" ]]; then
    if [[ ! -f "$explicit" ]]; then
      echo "$label does not exist: $explicit" >&2
      return 2
    fi
    realpath "$explicit"
    return
  fi
  local matches=()
  mapfile -d '' matches < <(
    find "$DPVO_DATA_ROOT" -maxdepth 7 -type f \
      -ipath "*quad*${token}*" \
      \( -iname '*ground*truth*.csv' -o -iname 'groundtruth.csv' -o -iname 'gt.csv' \) \
      -print0
  )
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "expected one $label under $DPVO_DATA_ROOT, found ${#matches[@]}; set its DPVO_NCD_* environment variable" >&2
    printf '%s\n' "${matches[@]}" >&2
    return 2
  fi
  realpath "${matches[0]}"
}

resolve_calibration() {
  local explicit="$1"
  if [[ -n "$explicit" ]]; then
    if [[ ! -f "$explicit" ]]; then
      echo "cam0 calibration does not exist: $explicit" >&2
      return 2
    fi
    realpath "$explicit"
    return
  fi
  local matches=()
  mapfile -d '' matches < <(
    find "$DPVO_DATA_ROOT" -maxdepth 7 -type f \
      \( -iname '*camchain*.yaml' -o -iname '*camchain*.yml' \) -print0
  )
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "expected one Collection 1 Kalibr camchain under $DPVO_DATA_ROOT, found ${#matches[@]}; set DPVO_NCD_CALIB" >&2
    printf '%s\n' "${matches[@]}" >&2
    return 2
  fi
  realpath "${matches[0]}"
}

BAG_EASY="$(resolve_bag 'Quad-Easy ROS1 bag' "${DPVO_NCD_EASY_BAG:-}" easy)"
BAG_HARD="$(resolve_bag 'Quad-Hard ROS1 bag' "${DPVO_NCD_HARD_BAG:-}" hard)"
GT_EASY="$(resolve_groundtruth 'Quad-Easy ground truth' "${DPVO_NCD_EASY_GROUNDTRUTH:-}" easy)"
GT_HARD="$(resolve_groundtruth 'Quad-Hard ground truth' "${DPVO_NCD_HARD_GROUNDTRUTH:-}" hard)"
CALIB="$(resolve_calibration "${DPVO_NCD_CALIB:-}")"
BAGS=("$BAG_EASY" "$BAG_HARD")
GROUNDTRUTH=("$GT_EASY" "$GT_HARD")
if [[ "$SCENARIO" == "three" ]]; then
  BAG_MEDIUM="$(resolve_bag 'Quad-Medium ROS1 bag' "${DPVO_NCD_MEDIUM_BAG:-}" medium)"
  GT_MEDIUM="$(resolve_groundtruth 'Quad-Medium ground truth' "${DPVO_NCD_MEDIUM_GROUNDTRUTH:-}" medium)"
  BAGS+=("$BAG_MEDIUM")
  GROUNDTRUTH+=("$GT_MEDIUM")
fi

for required in "$CALIB" "$DPVO_VOCAB" "$DPVO_DEPLOY_ROOT/dpvo.pth" "$DPVO_DEPLOY_ROOT/config/default.yaml"; do
  if [[ ! -f "$required" ]]; then
    echo "required input does not exist: $required" >&2
    exit 2
  fi
done

PYTHON=(pixi run --manifest-path "$DPVO_PIXI_MANIFEST" python)
STAGED_HELPER="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/newer_college_staged.py"

expected_for_robot() {
  case "$1" in
    robot0) echo "${DPVO_NCD_EXPECTED_EASY_FRAMES:-2986}" ;;
    robot1) echo "${DPVO_NCD_EXPECTED_HARD_FRAMES:-2821}" ;;
    *) echo "" ;;
  esac
}

setup_ros() {
  unset COLCON_CURRENT_PREFIX
  # ROS environment hooks are not nounset-safe.
  set +u
  source /opt/ros/jazzy/setup.bash
  source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"
  set -u
}

artifact_ready() {
  local robot_index="$1"
  local robot_id="${ROBOT_IDS[$robot_index]}"
  local expected
  expected="$(expected_for_robot "$robot_id")"
  local args=(
    artifact-ready
    --tracking-root "$ARTIFACT_ROOT"
    --robot-id "$robot_id"
    --rrd "$RRD_ROOT/$robot_id.rrd"
  )
  if [[ -n "$expected" ]]; then
    args+=(--expected-frames "$expected" --tolerance "${DPVO_NCD_FRAME_COUNT_TOLERANCE:-2}")
  fi
  "${PYTHON[@]}" "$STAGED_HELPER" "${args[@]}"
}

inspect_missing_sources() {
  for robot_index in "${!ROBOT_IDS[@]}"; do
    local robot_id="${ROBOT_IDS[$robot_index]}"
    local artifact_manifest="$ARTIFACT_ROOT/$robot_id/manifest.json"
    if [[ -f "$artifact_manifest" ]]; then
      artifact_ready "$robot_index"
      continue
    fi
    if [[ -e "$RRD_ROOT/$robot_id.rrd" ]]; then
      echo "refusing to overwrite orphan tracking RRD: $RRD_ROOT/$robot_id.rrd" >&2
      exit 2
    fi
    "${PYTHON[@]}" "$STAGED_HELPER" inspect-bag \
      --bag "${BAGS[$robot_index]}" \
      --calibration "$CALIB" \
      --topic "$SOURCE_TOPIC" \
      --stride 2 \
      --balance 0.0 \
      --output "$ARTIFACT_ROOT/source_inventory_${robot_id}.json"
  done
}

record_manifest() {
  local args=(
    record-manifest
    --run-dir "$RUN_DIR"
    --scenario "$SCENARIO"
    --calibration "$CALIB"
    --vocabulary "$DPVO_VOCAB"
    --network "$DPVO_DEPLOY_ROOT/dpvo.pth"
    --config "$DPVO_DEPLOY_ROOT/config/default.yaml"
  )
  for robot_index in "${!ROBOT_IDS[@]}"; do
    local robot_id="${ROBOT_IDS[$robot_index]}"
    args+=(
      --source "$robot_id=${BAGS[$robot_index]}"
      --groundtruth "$robot_id=${GROUNDTRUTH[$robot_index]}"
    )
    if [[ -f "$ARTIFACT_ROOT/source_inventory_${robot_id}.json" ]]; then
      args+=(--inventory "$robot_id=$ARTIFACT_ROOT/source_inventory_${robot_id}.json")
    fi
  done
  local parameters=(
    "tracking.config=$DPVO_DEPLOY_ROOT/config/default.yaml"
    "tracking.stride=2"
    "tracking.image_scale=1.0"
    "tracking.random_seed=1234"
    "tracking.camera_key=cam0"
    "tracking.source_topic=$SOURCE_TOPIC"
    "tracking.start_frame=0"
    "tracking.full_run_max_frames=0"
    "tracking.classic_intra_robot_loop_closure=true"
    "tracking.max_edge_age=48"
    "tracking.classic_bow_threshold=0.01"
    "tracking.inter_robot_retrieval=false"
    "tracking.inter_robot_optimization=false"
    "tracking.execution=sequential"
    "tracking.cuda_mps=false"
    "tracking.preflight_frames=300"
    "tracking.rectification_balance=0.0"
    "stage2.retrieval=dbow2_top1"
    "stage2.retrieval_rank=1"
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
    "stage2.teaser_required=true"
    "stage2.anchor_robot=robot0"
    "stage3.iterations=1000"
    "stage3.mode=alternating"
    "stage3.anchor_start_iteration=0"
    "stage3.anchor_stage_probability=0.5"
    "stage3.pose_warmup_iterations=0"
    "stage3.pose_block_iterations=20"
    "stage3.anchor_block_iterations=20"
    "stage3.target_hellinger=0.1"
    "stage3.contract_alpha=0.95"
    "stage3.d_reset=0.6"
    "stage3.odom_scale_sigma=-1.0"
    "stage3.inter_loop_scale_sigma=-1.0"
    "stage3.random_seed=42"
    "stage3.huber_k=-1.0"
    "stage3.centralized_max_iterations=300"
    "stage3.run_cbs=true"
    "stage3.run_centralized=true"
    "stage3.run_explicit_anchor_centralized=true"
    "stage3.write_rerun_rrd=true"
    "stage3.rerun_stream=false"
    "evaluation.pose_relation=trans_part"
    "evaluation.t_max_diff=0.055"
    "evaluation.align=true"
    "evaluation.correct_scale=true"
    "evaluation.reported_baseline=centralized_explicit_anchors"
    "plots.sparse_warp_mode=posewise"
    "plots.host_distance_quantile=0.90"
    "plots.projection_axes=xy"
    "plots.max_time_difference=0.055"
  )
  for parameter in "${parameters[@]}"; do
    args+=(--parameter "$parameter")
  done
  "${PYTHON[@]}" "$STAGED_HELPER" "${args[@]}"
}

run_preflight() {
  if [[ -f "$ARTIFACT_ROOT/robot0/manifest.json" ]]; then
    return
  fi
  local preflight_root="$RUN_DIR/preflight/easy/tracking"
  local preflight_manifest="$preflight_root/robot0/manifest.json"
  if [[ -f "$preflight_manifest" ]]; then
    "${PYTHON[@]}" "$STAGED_HELPER" artifact-ready \
      --tracking-root "$preflight_root" --robot-id robot0 \
      --expected-frames 300 --tolerance 0
    return
  fi
  if [[ -d "$preflight_root/robot0" ]]; then
    echo "incomplete Easy preflight exists; preserving it and stopping: $preflight_root/robot0" >&2
    exit 2
  fi
  mkdir -p "$preflight_root"
  echo "[preflight] 300 rectified Quad-Easy frames" | tee "$RUN_DIR/preflight/easy/preflight.log"
  ros2 launch dpvo_multi_robot single_robot_newer_college_track.launch.py \
    bag:="$BAG_EASY" \
    calib:="$CALIB" \
    robot_id:=robot0 \
    network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
    config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
    orb_vocab:="$DPVO_VOCAB" \
    tracking_artifact_output:="$preflight_root" \
    source_topic:="$SOURCE_TOPIC" \
    stride:=2 \
    image_scale:=1.0 \
    max_frames:=300 \
    random_seed:=1234 \
    local_feature_backend:=disk \
    bow_threshold:=0.01 \
    rectification_balance:=0.0 \
    2>&1 | tee -a "$RUN_DIR/preflight/easy/preflight.log"
  "${PYTHON[@]}" "$STAGED_HELPER" artifact-ready \
    --tracking-root "$preflight_root" --robot-id robot0 \
    --expected-frames 300 --tolerance 0
}

track_robot() {
  local robot_index="$1"
  local robot_id="${ROBOT_IDS[$robot_index]}"
  if [[ -f "$ARTIFACT_ROOT/$robot_id/manifest.json" ]]; then
    echo "[stage1] Reusing immutable $robot_id tracking artifact"
    artifact_ready "$robot_index"
    return
  fi
  if [[ -d "$ARTIFACT_ROOT/$robot_id" ]]; then
    echo "incomplete $robot_id artifact exists; preserving it and stopping: $ARTIFACT_ROOT/$robot_id" >&2
    exit 2
  fi
  echo "[stage1] Sequential tracking $robot_id from ${BAGS[$robot_index]}"
  ros2 launch dpvo_multi_robot single_robot_newer_college_track.launch.py \
    bag:="${BAGS[$robot_index]}" \
    calib:="$CALIB" \
    robot_id:="$robot_id" \
    network:="$DPVO_DEPLOY_ROOT/dpvo.pth" \
    config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
    orb_vocab:="$DPVO_VOCAB" \
    tracking_artifact_output:="$ARTIFACT_ROOT" \
    source_topic:="$SOURCE_TOPIC" \
    stride:=2 \
    image_scale:=1.0 \
    max_frames:=0 \
    random_seed:=1234 \
    local_feature_backend:=disk \
    bow_threshold:=0.01 \
    rectification_balance:=0.0 \
    rerun_save:="$RRD_ROOT/$robot_id.rrd" \
    rerun_recording_id:="newer-college-$robot_id-tracking" \
    2>&1 | tee "$RUN_DIR/stage1_${robot_id}.log"
  artifact_ready "$robot_index"
}

validate_stage1() {
  local args=(
    validate-stage1
    --tracking-root "$ARTIFACT_ROOT"
    --rrd-root "$RRD_ROOT"
    --tracking-graph "$ARTIFACT_ROOT/unoptimized_tracking_graph_${SCENARIO}.json"
    --robot-ids "${ROBOT_IDS[@]}"
    --tolerance "${DPVO_NCD_FRAME_COUNT_TOLERANCE:-2}"
    --output "$STAGE1_GATE"
    --expected "robot0=${DPVO_NCD_EXPECTED_EASY_FRAMES:-2986}"
    --expected "robot1=${DPVO_NCD_EXPECTED_HARD_FRAMES:-2821}"
  )
  "${PYTHON[@]}" "$STAGED_HELPER" "${args[@]}"
}

run_stage1() {
  echo "[stage1] Newer College $SCENARIO-robot tracking; no inter-robot retrieval or optimization"
  inspect_missing_sources
  record_manifest
  setup_ros
  run_preflight
  for robot_index in "${!ROBOT_IDS[@]}"; do
    track_robot "$robot_index"
  done
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_multirobot assemble \
    --artifact-root "$ARTIFACT_ROOT" \
    --robot-ids "${ROBOT_IDS[@]}" \
    --output-base "$ARTIFACT_ROOT/unoptimized_tracking_graph_${SCENARIO}" \
    --anchor-robot robot0 \
    --odometry-weight 100.0
  validate_stage1
}

check_graph() {
  "${PYTHON[@]}" "$STAGED_HELPER" check-graph \
    --graph "$GV_DIR/unoptimized_verified_graph.json" \
    --robot-ids "${ROBOT_IDS[@]}" \
    --min-pair robot0:robot1=4 \
    --output "$GRAPH_GATE"
}

run_stage2() {
  echo "[stage2] $SCENARIO: DBoW2 top-1 -> DISK/LightGlue -> TEASER++"
  validate_stage1
  if [[ -s "$GV_DIR/verification.json" && -s "$GV_DIR/unoptimized_verified_graph.json" && -s "$GV_DIR/unoptimized_verified_graph.g2o" ]]; then
    echo "[stage2] Reusing completed geometric verification in $GV_DIR"
    check_graph
    return
  fi
  if [[ -d "$GV_DIR" ]] && find "$GV_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial stage-two directory exists; preserving diagnostics and stopping: $GV_DIR" >&2
    exit 2
  fi
  mkdir -p "$GV_DIR"
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_multirobot verify \
    --artifact-root "$ARTIFACT_ROOT" \
    --robot-ids "${ROBOT_IDS[@]}" \
    --orb-vocab "$DPVO_VOCAB" \
    --output-dir "$GV_DIR" \
    --anchor-robot robot0 \
    --local-feature-backend disk \
    --bow-threshold 0.01 \
    --bow-repetitions 1 \
    --bow-nms-radius 10 \
    --teaser-noise-bound 0.10 \
    --min-inliers 15 \
    --min-inlier-ratio 0.15 \
    --max-depth 20.0 \
    --odometry-weight 100.0 \
    --random-seed 1234 \
    2>&1 | tee "$RUN_DIR/stage2_${SCENARIO}.log"
  check_graph
}

run_stage3() {
  echo "[stage3] $SCENARIO: byte-identical raw graph -> CBS and both centralized parameterizations"
  validate_stage1
  check_graph
  if [[ -f "$DPGO_DIR/offline_dpgo_provenance.json" ]]; then
    if "${PYTHON[@]}" -c '
import hashlib, json, pathlib, sys
provenance_path, source_path, copied_path, output_dir = map(pathlib.Path, sys.argv[1:])
p = json.loads(provenance_path.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
required_outputs = (
    "input_keyframes_unoptimized.g2o",
    "centralized.csv",
    "centralized_explicit_anchors.csv",
    "cbs.csv",
    "dpvo_sim3_cbs.rrd",
)
valid = (
    p.get("status") == "complete"
    and p.get("return_code") == 0
    and p.get("copied_input_is_byte_identical") is True
    and source_path.is_file()
    and copied_path.is_file()
    and digest(source_path) == digest(copied_path)
    and p.get("source_sha256") == digest(source_path)
    and p.get("copied_input_sha256") == digest(copied_path)
    and all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in required_outputs)
)
raise SystemExit(0 if valid else 1)
' "$DPGO_DIR/offline_dpgo_provenance.json" \
      "$GV_DIR/unoptimized_verified_graph.json" \
      "$DPGO_DIR/input_keyframes_unoptimized.json" \
      "$DPGO_DIR"; then
      echo "[stage3] Reusing completed solver outputs in $DPGO_DIR"
      return
    fi
    echo "previous stage-three attempt is not complete; preserving diagnostics and stopping" >&2
    exit 2
  fi
  if [[ -d "$DPGO_DIR" ]] && find "$DPGO_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial stage-three directory has no provenance; preserving it and stopping: $DPGO_DIR" >&2
    exit 2
  fi
  mkdir -p "$DPGO_DIR"
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_dpgo \
    --input-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --output-dir "$DPGO_DIR" \
    --cbs-executable "${DPVO_CBS_EXECUTABLE:-$DPVO_DEPLOY_ROOT/build/cbs/examples/cbs_dpvo_sim3_offline}" \
    --iterations 1000 \
    --stage-mode alternating \
    --anchor-start-iteration 0 \
    --anchor-stage-probability 0.5 \
    --pose-warmup-iterations 0 \
    --pose-block-iterations 20 \
    --anchor-block-iterations 20 \
    --target-hellinger 0.1 \
    --contract-alpha 0.95 \
    --d-reset 0.6 \
    --odom-scale-sigma -1.0 \
    --inter-loop-scale-sigma -1.0 \
    --huber-k -1.0 \
    --centralized-max-iterations 300 \
    --random-seed 42 \
    --write-rerun-rrd \
    2>&1 | tee "$RUN_DIR/stage3_${SCENARIO}.log"
}

run_evaluate() {
  echo "[evaluate] Base ground truth -> cam0; evo Sim(3) alignment at 55 ms tolerance"
  validate_stage1
  check_graph
  if [[ -f "$METRICS" ]]; then
    echo "[evaluate] Reusing completed evo outputs in $EVO_DIR"
    "${PYTHON[@]}" "$STAGED_HELPER" final-gate \
      --stage1-gate "$STAGE1_GATE" \
      --graph-gate "$GRAPH_GATE" \
      --verified-graph "$GV_DIR/unoptimized_verified_graph.json" \
      --dpgo-dir "$DPGO_DIR" \
      --evo-dir "$EVO_DIR" \
      --metrics "$METRICS" \
      --robot-ids "${ROBOT_IDS[@]}" \
      --output "$FINAL_GATE"
    return
  fi
  if [[ -d "$EVO_DIR" ]] && find "$EVO_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial evaluation directory exists; preserving diagnostics and stopping: $EVO_DIR" >&2
    exit 2
  fi
  mkdir -p "$EVO_DIR/results"
  local gt_args=()
  for robot_index in "${!ROBOT_IDS[@]}"; do
    gt_args+=(--groundtruth "${ROBOT_IDS[$robot_index]}=${GROUNDTRUTH[$robot_index]}")
  done
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/export_newer_college_evo_tum.py" \
    --graph "$GV_DIR/unoptimized_verified_graph.json" \
    --solver-dir "$DPGO_DIR" \
    --calibration "$CALIB" \
    "${gt_args[@]}" \
    --output-dir "$EVO_DIR"

  local solutions=(centralized centralized_explicit_anchors cbs)
  local scopes=("${ROBOT_IDS[@]}" joint)
  for solution in "${solutions[@]}"; do
    for scope in "${scopes[@]}"; do
      pixi run --manifest-path "$DPVO_PIXI_MANIFEST" evo_ape tum \
        "$EVO_DIR/groundtruth_${scope}.tum" \
        "$EVO_DIR/${solution}_${scope}.tum" \
        --pose_relation trans_part \
        --align \
        --correct_scale \
        --t_max_diff 0.055 \
        --save_results "$EVO_DIR/results/${solution}_${scope}.zip" \
        --no_warnings
    done
  done
  "${PYTHON[@]}" "$STAGED_HELPER" collect-metrics \
    --evo-dir "$EVO_DIR" \
    --robot-ids "${ROBOT_IDS[@]}" \
    --output "$METRICS"
  "${PYTHON[@]}" "$STAGED_HELPER" final-gate \
    --stage1-gate "$STAGE1_GATE" \
    --graph-gate "$GRAPH_GATE" \
    --verified-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --dpgo-dir "$DPGO_DIR" \
    --evo-dir "$EVO_DIR" \
    --metrics "$METRICS" \
    --robot-ids "${ROBOT_IDS[@]}" \
    --output "$FINAL_GATE"
}

run_plot() {
  echo "[plot] Trajectory/loop and sparse joint-map figures"
  "${PYTHON[@]}" "$STAGED_HELPER" final-gate \
    --stage1-gate "$STAGE1_GATE" \
    --graph-gate "$GRAPH_GATE" \
    --verified-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --dpgo-dir "$DPGO_DIR" \
    --evo-dir "$EVO_DIR" \
    --metrics "$METRICS" \
    --robot-ids "${ROBOT_IDS[@]}" \
    --output "$FINAL_GATE"
  if [[ -f "$PLOTS_DIR/newer_college_trajectories_loops.png" && -f "$PLOTS_DIR/newer_college_sparse_joint_map_alignment.png" ]]; then
    echo "[plot] Reusing completed plots in $PLOTS_DIR"
    "${PYTHON[@]}" "$STAGED_HELPER" final-gate \
      --stage1-gate "$STAGE1_GATE" \
      --graph-gate "$GRAPH_GATE" \
      --verified-graph "$GV_DIR/unoptimized_verified_graph.json" \
      --dpgo-dir "$DPGO_DIR" \
      --evo-dir "$EVO_DIR" \
      --metrics "$METRICS" \
      --plots-dir "$PLOTS_DIR" \
      --robot-ids "${ROBOT_IDS[@]}" \
      --output "$FINAL_GATE"
    return
  fi
  if [[ -d "$PLOTS_DIR" ]] && find "$PLOTS_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial plot directory exists; preserving diagnostics and stopping: $PLOTS_DIR" >&2
    exit 2
  fi
  mkdir -p "$PLOTS_DIR"
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/plot_newer_college.py" \
    --graph "$GV_DIR/unoptimized_verified_graph.json" \
    --solver-dir "$DPGO_DIR" \
    --evo-dir "$EVO_DIR" \
    --output-dir "$PLOTS_DIR"

  local npz_args=()
  for robot_id in "${ROBOT_IDS[@]}"; do
    npz_args+=(--input-npz "$robot_id=$ARTIFACT_ROOT/$robot_id/map.npz")
  done
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/visualize_cbs_joint_map.py" \
    "${npz_args[@]}" \
    --input-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --cbs-csv "$DPGO_DIR/cbs.csv" \
    --comparison-csv "Explicit-anchor centralized=$DPGO_DIR/centralized_explicit_anchors.csv" \
    --output-rrd "$PLOTS_DIR/cbs_sparse_joint_map.rrd" \
    --output-ply "$PLOTS_DIR/cbs_sparse_joint_map.ply" \
    --output-manifest "$PLOTS_DIR/cbs_sparse_joint_map.json" \
    --recording-id "newer-college-$SCENARIO-cbs-sparse-map" \
    --host-distance-quantile 0.90 \
    --warp-mode posewise
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/visualize_cbs_joint_map.py" \
    "${npz_args[@]}" \
    --input-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --cbs-csv "$DPGO_DIR/centralized_explicit_anchors.csv" \
    --primary-label "Explicit-anchor centralized" \
    --comparison-csv "CBS=$DPGO_DIR/cbs.csv" \
    --output-rrd "$PLOTS_DIR/centralized_sparse_joint_map.rrd" \
    --output-ply "$PLOTS_DIR/centralized_sparse_joint_map.ply" \
    --output-manifest "$PLOTS_DIR/centralized_sparse_joint_map.json" \
    --recording-id "newer-college-$SCENARIO-centralized-sparse-map" \
    --host-distance-quantile 0.90 \
    --warp-mode posewise
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/plot_joint_map_ply.py" \
    --centralized-ply "$PLOTS_DIR/centralized_sparse_joint_map.ply" \
    --cbs-ply "$PLOTS_DIR/cbs_sparse_joint_map.ply" \
    --centralized-csv "$DPGO_DIR/centralized_explicit_anchors.csv" \
    --cbs-csv "$DPGO_DIR/cbs.csv" \
    --evo-dir "$EVO_DIR" \
    --centralized-solution-name centralized_explicit_anchors \
    --cbs-solution-name cbs \
    --output-dir "$PLOTS_DIR" \
    --prefix newer_college_sparse \
    --projection-axes xy \
    --max-time-difference 0.055
  "${PYTHON[@]}" "$STAGED_HELPER" final-gate \
    --stage1-gate "$STAGE1_GATE" \
    --graph-gate "$GRAPH_GATE" \
    --verified-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --dpgo-dir "$DPGO_DIR" \
    --evo-dir "$EVO_DIR" \
    --metrics "$METRICS" \
    --plots-dir "$PLOTS_DIR" \
    --robot-ids "${ROBOT_IDS[@]}" \
    --output "$FINAL_GATE"
}

case "$STAGE" in
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  stage3) run_stage3 ;;
  evaluate) run_evaluate ;;
  plot) run_plot ;;
  all)
    run_stage1
    run_stage2
    run_stage3
    run_evaluate
    run_plot
    ;;
esac

echo "Completed Newer College $STAGE ($SCENARIO robots): $RUN_DIR"

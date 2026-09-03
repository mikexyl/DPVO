#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <stage1|stage2|stage3|evaluate|plot|all> <two|four>" >&2
  exit 2
fi
STAGE="$1"
SCENARIO="$2"
case "$STAGE" in
  stage1|stage2|stage3|evaluate|plot|all) ;;
  *) echo "invalid stage: $STAGE" >&2; exit 2 ;;
esac
case "$SCENARIO" in
  two)
    ROBOT_IDS=(robot0 robot1)
    SOURCE_ROBOTS=(robot1 robot2)
    EXPECTED_FRAMES=(4999 5575)
    ;;
  four)
    ROBOT_IDS=(robot0 robot1 robot2 robot3)
    SOURCE_ROBOTS=(robot1 robot2 robot3 robot4)
    EXPECTED_FRAMES=(4999 5575 10489 13887)
    ;;
  *) echo "invalid scenario: $SCENARIO" >&2; exit 2 ;;
esac

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/data3/dpvo_cbs_ws/src/DPVO}"
SOURCE_ROOT="${DPVO_CU_SOURCE_ROOT:-/mnt/data/cu-multi/main_campus}"
WORK_ROOT="${DPVO_CU_WORK_ROOT:-/data3/mikexyl/datasets/cu-multi/main_campus}"
RUN_DIR="${DPVO_CU_RESULT_ROOT:-/data3/mikexyl/results/dpvo_multi_robot/cu_multi_main_campus_staged_20260902}"
DPVO_VOCAB="${DPVO_ORB_VOCAB:-/data3/mikexyl/datasets/orb_vocab/ORBvoc.txt}"
PIXI_MANIFEST="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml"
CBS_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"
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
GATE_HELPER="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/newer_college_staged.py"
MANIFEST_HELPER="$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/cu_multi_staged.py"

export PATH="/home/mikexyl/.pixi/bin:$PATH"
export DPVO_ROOT="$DPVO_DEPLOY_ROOT"
export DPVO_PIXI_MANIFEST="$PIXI_MANIFEST"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-74}"
export ROS_LOG_DIR
export PYTHONWARNINGS="${DPVO_PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning}"
export LD_LIBRARY_PATH="$CBS_PREFIX/gtsam/lib:$CBS_PREFIX/aria_common/lib:$CBS_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT" "$RRD_ROOT" "$ROS_LOG_DIR"
PYTHON=(pixi run --manifest-path "$PIXI_MANIFEST" python)
BAGS=()
ARCHIVES=()
GROUNDTRUTH=()
for source_robot in "${SOURCE_ROBOTS[@]}"; do
  BAGS+=("$WORK_ROOT/$source_robot/${source_robot}_main_campus_camera_rgb")
  ARCHIVES+=("$SOURCE_ROOT/$source_robot/${source_robot}_main_campus_camera_rgb.zip")
  GROUNDTRUTH+=("$SOURCE_ROOT/$source_robot/${source_robot}_main_campus_gt_utm_poses.csv")
done
for path in "$DPVO_VOCAB" "$DPVO_DEPLOY_ROOT/dpvo.pth" "$DPVO_DEPLOY_ROOT/config/default.yaml" \
  "${BAGS[@]}" "${ARCHIVES[@]}" "${GROUNDTRUTH[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done

setup_ros() {
  unset COLCON_CURRENT_PREFIX
  set +u
  source /opt/ros/jazzy/setup.bash
  source "$DPVO_DEPLOY_ROOT/build/ros2-install/setup.bash"
  set -u
}

artifact_ready() {
  local index="$1"
  local robot_id="${ROBOT_IDS[$index]}"
  "${PYTHON[@]}" "$GATE_HELPER" artifact-ready \
    --tracking-root "$ARTIFACT_ROOT" \
    --robot-id "$robot_id" \
    --expected-frames "${EXPECTED_FRAMES[$index]}" \
    --tolerance 0 \
    --rrd "$RRD_ROOT/$robot_id.rrd"
}

record_manifest() {
  local args=(
    --run-dir "$RUN_DIR"
    --scenario "$SCENARIO"
    --vocabulary "$DPVO_VOCAB"
    --network "$DPVO_DEPLOY_ROOT/dpvo.pth"
    --config "$DPVO_DEPLOY_ROOT/config/default.yaml"
  )
  for index in "${!ROBOT_IDS[@]}"; do
    args+=(
      --source-archive "${ROBOT_IDS[$index]}=${ARCHIVES[$index]}"
      --bag "${ROBOT_IDS[$index]}=${BAGS[$index]}"
      --groundtruth "${ROBOT_IDS[$index]}=${GROUNDTRUTH[$index]}"
    )
  done
  local parameters=(
    "tracking.config=$DPVO_DEPLOY_ROOT/config/default.yaml"
    "tracking.stride=2"
    "tracking.source_rate_hz=10.0"
    "tracking.image_scale=1.0"
    "tracking.random_seed=1234"
    "tracking.start_frame=0"
    "tracking.full_run_max_frames=0"
    "tracking.rectification_alpha=0.0"
    "tracking.classic_intra_robot_loop_closure=true"
    "tracking.inter_robot_retrieval=false"
    "tracking.inter_robot_optimization=false"
    "tracking.execution=sequential"
    "tracking.preflight_start_frame_after_stride=300"
    "tracking.preflight_frames=300"
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
    "stage3.iterations=1000"
    "stage3.mode=alternating"
    "stage3.pose_block_iterations=20"
    "stage3.anchor_block_iterations=20"
    "stage3.target_hellinger=0.1"
    "stage3.d_reset=0.6"
    "stage3.random_seed=42"
    "stage3.huber_k=-1.0"
    "stage3.centralized_max_iterations=300"
    "evaluation.groundtruth_frame=UTM_base_link_proxy"
    "evaluation.camera_extrinsic_applied=false"
    "evaluation.align=true"
    "evaluation.correct_scale=true"
    "evaluation.t_max_diff=0.055"
    "evaluation.reported_baseline=centralized_explicit_anchors"
  )
  for parameter in "${parameters[@]}"; do args+=(--parameter "$parameter"); done
  "${PYTHON[@]}" "$MANIFEST_HELPER" "${args[@]}"
}

run_preflight() {
  if [[ -f "$ARTIFACT_ROOT/robot0/manifest.json" ]]; then return; fi
  local root="$RUN_DIR/preflight/robot1_moving_stride2"
  if [[ -f "$root/robot0/manifest.json" ]]; then
    "${PYTHON[@]}" "$GATE_HELPER" artifact-ready \
      --tracking-root "$root" --robot-id robot0 --expected-frames 300 --tolerance 0
    return
  fi
  if [[ -d "$root/robot0" ]]; then
    echo "incomplete preflight exists; preserving it: $root/robot0" >&2
    exit 2
  fi
  ros2 launch dpvo_multi_robot single_robot_cu_multi_track.launch.py \
    bag:="${BAGS[0]}" source_robot:=robot1 robot_id:=robot0 \
    network:="$DPVO_DEPLOY_ROOT/dpvo.pth" config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
    orb_vocab:="$DPVO_VOCAB" tracking_artifact_output:="$root" \
    stride:=2 start_frame:=300 max_frames:=300 image_scale:=1.0 \
    rectification_alpha:=0.0 random_seed:=1234 local_feature_backend:=disk \
    bow_threshold:=0.01 2>&1 | tee "$RUN_DIR/preflight_robot1_moving_stride2.log"
  "${PYTHON[@]}" "$GATE_HELPER" artifact-ready \
    --tracking-root "$root" --robot-id robot0 --expected-frames 300 --tolerance 0
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
  ros2 launch dpvo_multi_robot single_robot_cu_multi_track.launch.py \
    bag:="${BAGS[$index]}" source_robot:="${SOURCE_ROBOTS[$index]}" robot_id:="$robot_id" \
    network:="$DPVO_DEPLOY_ROOT/dpvo.pth" config:="$DPVO_DEPLOY_ROOT/config/default.yaml" \
    orb_vocab:="$DPVO_VOCAB" tracking_artifact_output:="$ARTIFACT_ROOT" \
    stride:=2 start_frame:=0 max_frames:=0 image_scale:=1.0 rectification_alpha:=0.0 \
    random_seed:=1234 local_feature_backend:=disk bow_threshold:=0.01 \
    rerun_save:="$RRD_ROOT/$robot_id.rrd" \
    rerun_recording_id:="cu-multi-main-campus-$robot_id-tracking" \
    2>&1 | tee "$RUN_DIR/stage1_${robot_id}.log"
  artifact_ready "$index"
}

validate_stage1() {
  local args=(validate-stage1 --tracking-root "$ARTIFACT_ROOT" --rrd-root "$RRD_ROOT"
    --tracking-graph "$ARTIFACT_ROOT/unoptimized_tracking_graph_${SCENARIO}.json"
    --robot-ids "${ROBOT_IDS[@]}" --tolerance 0 --output "$STAGE1_GATE")
  for index in "${!ROBOT_IDS[@]}"; do
    args+=(--expected "${ROBOT_IDS[$index]}=${EXPECTED_FRAMES[$index]}")
  done
  "${PYTHON[@]}" "$GATE_HELPER" "${args[@]}"
  local diagnostic_args=()
  for index in "${!ROBOT_IDS[@]}"; do
    diagnostic_args+=(
      --artifact "${ROBOT_IDS[$index]}=$ARTIFACT_ROOT/${ROBOT_IDS[$index]}"
      --groundtruth "${ROBOT_IDS[$index]}=${GROUNDTRUTH[$index]}"
    )
  done
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/analyze_cu_multi_tracking.py" \
    "${diagnostic_args[@]}" --max-time-difference 0.055 \
    --output "$ARTIFACT_ROOT/raw_tracking_diagnostics_${SCENARIO}.json"
}

run_stage1() {
  echo "[stage1] CU-Multi Main Campus $SCENARIO scenario"
  record_manifest
  setup_ros
  run_preflight
  for index in "${!ROBOT_IDS[@]}"; do track_robot "$index"; done
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_multirobot assemble \
    --artifact-root "$ARTIFACT_ROOT" --robot-ids "${ROBOT_IDS[@]}" \
    --output-base "$ARTIFACT_ROOT/unoptimized_tracking_graph_${SCENARIO}" \
    --anchor-robot robot0 --odometry-weight 100.0
  validate_stage1
}

check_graph() {
  "${PYTHON[@]}" "$GATE_HELPER" check-graph \
    --graph "$GV_DIR/unoptimized_verified_graph.json" --robot-ids "${ROBOT_IDS[@]}" \
    --min-pair robot0:robot1=4 --output "$GRAPH_GATE"
}

run_stage2() {
  validate_stage1
  if [[ -s "$GV_DIR/verification.json" && -s "$GV_DIR/unoptimized_verified_graph.json" && -s "$GV_DIR/unoptimized_verified_graph.g2o" ]]; then
    check_graph
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
    --random-seed 1234 2>&1 | tee "$RUN_DIR/stage2_${SCENARIO}.log"
  check_graph
}

run_stage3() {
  validate_stage1
  check_graph
  if [[ -f "$DPGO_DIR/offline_dpgo_provenance.json" ]]; then
    "${PYTHON[@]}" -c '
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if p.get("status")=="complete" and p.get("return_code")==0 and p.get("copied_input_is_byte_identical") is True else 1)
' "$DPGO_DIR/offline_dpgo_provenance.json" || {
      echo "incomplete stage-three provenance; preserving diagnostics" >&2; exit 2; }
    return
  fi
  if [[ -d "$DPGO_DIR" ]] && find "$DPGO_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial stage-three directory exists; preserving it: $DPGO_DIR" >&2
    exit 2
  fi
  mkdir -p "$DPGO_DIR"
  "${PYTHON[@]}" -m dpvo.loop_closure.offline_dpgo \
    --input-graph "$GV_DIR/unoptimized_verified_graph.json" --output-dir "$DPGO_DIR" \
    --cbs-executable "${DPVO_CBS_EXECUTABLE:-$DPVO_DEPLOY_ROOT/build/cbs/examples/cbs_dpvo_sim3_offline}" \
    --iterations 1000 --stage-mode alternating --anchor-start-iteration 0 \
    --anchor-stage-probability 0.5 --pose-warmup-iterations 0 \
    --pose-block-iterations 20 --anchor-block-iterations 20 \
    --target-hellinger 0.1 --contract-alpha 0.95 --d-reset 0.6 \
    --odom-scale-sigma -1.0 --inter-loop-scale-sigma -1.0 --huber-k -1.0 \
    --centralized-max-iterations 300 --random-seed 42 --write-rerun-rrd \
    2>&1 | tee "$RUN_DIR/stage3_${SCENARIO}.log"
}

final_gate() {
  local args=(final-gate --stage1-gate "$STAGE1_GATE" --graph-gate "$GRAPH_GATE"
    --verified-graph "$GV_DIR/unoptimized_verified_graph.json" --dpgo-dir "$DPGO_DIR"
    --evo-dir "$EVO_DIR" --metrics "$METRICS" --robot-ids "${ROBOT_IDS[@]}"
    --plot-prefix cu_multi --output "$FINAL_GATE")
  if [[ "${1:-}" == "plots" ]]; then args+=(--plots-dir "$PLOTS_DIR"); fi
  "${PYTHON[@]}" "$GATE_HELPER" "${args[@]}"
}

run_evaluate() {
  validate_stage1
  check_graph
  if [[ -f "$METRICS" ]]; then final_gate; return; fi
  if [[ -d "$EVO_DIR" ]] && find "$EVO_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial evaluation exists; preserving it: $EVO_DIR" >&2
    exit 2
  fi
  mkdir -p "$EVO_DIR/results"
  local gt_args=()
  for index in "${!ROBOT_IDS[@]}"; do
    gt_args+=(--groundtruth "${ROBOT_IDS[$index]}=${GROUNDTRUTH[$index]}")
  done
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/export_cu_multi_evo_tum.py" \
    --graph "$GV_DIR/unoptimized_verified_graph.json" --solver-dir "$DPGO_DIR" \
    "${gt_args[@]}" --output-dir "$EVO_DIR"
  local solutions=(centralized centralized_explicit_anchors cbs)
  local scopes=("${ROBOT_IDS[@]}" joint)
  for solution in "${solutions[@]}"; do
    for scope in "${scopes[@]}"; do
      pixi run --manifest-path "$PIXI_MANIFEST" evo_ape tum \
        "$EVO_DIR/groundtruth_${scope}.tum" "$EVO_DIR/${solution}_${scope}.tum" \
        --pose_relation trans_part --align --correct_scale --t_max_diff 0.055 \
        --save_results "$EVO_DIR/results/${solution}_${scope}.zip" --no_warnings
    done
  done
  "${PYTHON[@]}" "$GATE_HELPER" collect-metrics \
    --evo-dir "$EVO_DIR" --robot-ids "${ROBOT_IDS[@]}" --output "$METRICS"
  final_gate
}

run_plot() {
  final_gate
  if [[ -f "$PLOTS_DIR/cu_multi_trajectories_loops.png" && -f "$PLOTS_DIR/cu_multi_sparse_joint_map_alignment.png" ]]; then
    final_gate plots
    return
  fi
  if [[ -d "$PLOTS_DIR" ]] && find "$PLOTS_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "partial plot directory exists; preserving it: $PLOTS_DIR" >&2
    exit 2
  fi
  mkdir -p "$PLOTS_DIR"
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/plot_newer_college.py" \
    --graph "$GV_DIR/unoptimized_verified_graph.json" --solver-dir "$DPGO_DIR" \
    --evo-dir "$EVO_DIR" --output-dir "$PLOTS_DIR" \
    --dataset-label CU-Multi --output-prefix cu_multi
  local npz_args=()
  for robot_id in "${ROBOT_IDS[@]}"; do
    npz_args+=(--input-npz "$robot_id=$ARTIFACT_ROOT/$robot_id/map.npz")
  done
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/visualize_cbs_joint_map.py" \
    "${npz_args[@]}" --input-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --cbs-csv "$DPGO_DIR/cbs.csv" \
    --comparison-csv "Explicit-anchor centralized=$DPGO_DIR/centralized_explicit_anchors.csv" \
    --output-rrd "$PLOTS_DIR/cbs_sparse_joint_map.rrd" \
    --output-ply "$PLOTS_DIR/cbs_sparse_joint_map.ply" \
    --output-manifest "$PLOTS_DIR/cbs_sparse_joint_map.json" \
    --recording-id "cu-multi-$SCENARIO-cbs-sparse-map" \
    --host-distance-quantile 0.90 --warp-mode posewise
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/visualize_cbs_joint_map.py" \
    "${npz_args[@]}" --input-graph "$GV_DIR/unoptimized_verified_graph.json" \
    --cbs-csv "$DPGO_DIR/centralized_explicit_anchors.csv" \
    --primary-label "Explicit-anchor centralized" \
    --comparison-csv "CBS=$DPGO_DIR/cbs.csv" \
    --output-rrd "$PLOTS_DIR/centralized_sparse_joint_map.rrd" \
    --output-ply "$PLOTS_DIR/centralized_sparse_joint_map.ply" \
    --output-manifest "$PLOTS_DIR/centralized_sparse_joint_map.json" \
    --recording-id "cu-multi-$SCENARIO-centralized-sparse-map" \
    --host-distance-quantile 0.90 --warp-mode posewise
  "${PYTHON[@]}" "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/plot_joint_map_ply.py" \
    --centralized-ply "$PLOTS_DIR/centralized_sparse_joint_map.ply" \
    --cbs-ply "$PLOTS_DIR/cbs_sparse_joint_map.ply" \
    --centralized-csv "$DPGO_DIR/centralized_explicit_anchors.csv" \
    --cbs-csv "$DPGO_DIR/cbs.csv" --evo-dir "$EVO_DIR" \
    --centralized-solution-name centralized_explicit_anchors \
    --cbs-solution-name cbs --output-dir "$PLOTS_DIR" \
    --prefix cu_multi_sparse --projection-axes xy --max-time-difference 0.055
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

echo "Completed CU-Multi Main Campus $STAGE ($SCENARIO): $RUN_DIR"

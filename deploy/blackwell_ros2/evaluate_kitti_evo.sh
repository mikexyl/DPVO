#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 RESULT_DIR [DATASET_ROOT] [TAG]" >&2
  exit 2
fi

result_dir="$1"
dataset_root="${2:-/data1/mikexyl/datasets/kitti_odometry/dataset}"
tag="${3:-$(basename "$result_dir")}"
evo_dir="$result_dir/evo"
result_zip_dir="$evo_dir/results"
mkdir -p "$result_zip_dir"

pixi run python deploy/blackwell_ros2/export_kitti_evo_tum.py \
  "$result_dir" \
  --dataset-root "$dataset_root" \
  --tag "$tag" \
  --windows \
  "${DPVO_START0:-0}" "${DPVO_END0:-1450}" \
  "${DPVO_START1:-1450}" "${DPVO_END1:-3000}" \
  "${DPVO_START2:-3000}" "${DPVO_END2:-4541}"

solutions=(
  raw
  online_centralized
  full_graph_centralized
  cbs
)
scopes=(robot0 robot1 robot2 joint)

for solution in "${solutions[@]}"; do
  for scope in "${scopes[@]}"; do
    pixi run evo_ape tum \
      "$evo_dir/groundtruth_${scope}.tum" \
      "$evo_dir/${solution}_${scope}.tum" \
      --pose_relation trans_part \
      --align \
      --correct_scale \
      --t_max_diff 0.02 \
      --save_results "$result_zip_dir/${solution}_${scope}.zip" \
      --no_warnings
  done
done

pixi run python deploy/blackwell_ros2/plot_kitti_trajectories.py \
  "$result_dir" \
  --tag "$tag"

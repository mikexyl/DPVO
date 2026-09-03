#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RESULT_DIR [BAG_ROOT]" >&2
  exit 2
fi

result_dir="$1"
bag_root="${2:-/data/euroc}"
evo_dir="$result_dir/evo"
result_zip_dir="$evo_dir/results"
mkdir -p "$result_zip_dir"

pixi run python deploy/blackwell_ros2/export_euroc_evo_tum.py \
  "$result_dir" \
  --bag-root "$bag_root"

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

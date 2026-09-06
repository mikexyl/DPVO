#!/usr/bin/env bash
set -eo pipefail

export PATH="/home/mikexyl/.pixi/bin:$PATH"

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

overlap_frames="${DPVO_KITTI_OVERLAP_FRAMES:-200}"
sequence="${DPVO_KITTI_SEQUENCE:-00}"
case "$sequence:$overlap_frames" in
  00:200)
    windows=(0 554 354 1008 808 1462 1262 1916 1716 2370 2170 2825 2625 3279 3079 3733 3533 4187 3987 4541)
    ;;
  00:50)
    windows=(0 479 429 933 883 1387 1337 1841 1791 2295 2245 2750 2700 3204 3154 3658 3608 4112 4062 4541)
    ;;
  05:200)
    windows=(0 376 176 652 452 928 728 1204 1004 1480 1280 1757 1557 2033 1833 2309 2109 2585 2385 2761)
    ;;
  05:50)
    windows=(0 301 251 577 527 853 803 1129 1079 1405 1355 1682 1632 1958 1908 2234 2184 2510 2460 2761)
    ;;
  *)
    echo "unsupported KITTI sequence/overlap combination: $sequence/$overlap_frames" >&2
    exit 2
    ;;
esac

pixi run python deploy/blackwell_ros2/export_kitti_evo_tum.py \
  "$result_dir" \
  --dataset-root "$dataset_root" \
  --sequence "$sequence" \
  --tag "$tag" \
  --windows "${windows[@]}"

solutions=(raw full_graph_centralized cbs)
scopes=(robot0 robot1 robot2 robot3 robot4 robot5 robot6 robot7 robot8 robot9 joint)

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

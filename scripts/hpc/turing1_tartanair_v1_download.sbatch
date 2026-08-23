#!/bin/bash -l
#SBATCH --job-name=tartanair-v1
#SBATCH --partition=COMPUTE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export LANG=C
export LC_ALL=C
export PYTHONUNBUFFERED=1
export HF_HOME="${TARTANAIR_HF_HOME:-/mnt/beegfs/xliu0001/.cache/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"

project_dir="${DPVO_PROJECT_DIR:-/mnt/beegfs/xliu0001/DPVO}"
tools_dir="${TARTANAIR_TOOLS_DIR:-/mnt/beegfs/xliu0001/tools/tartanair_tools}"
output_dir="${TARTANAIR_OUTPUT_DIR:-/mnt/beegfs/xliu0001/datasets/TartanAir}"
python_bin="${TARTANAIR_PYTHON:-$tools_dir/.venv/bin/python}"

mkdir -p "$project_dir/slurm_logs" "$output_dir" "$HF_HOME"
cd "$tools_dir"

echo "timestamp=$(date --iso-8601=seconds)"
echo "host=$(hostname --fqdn 2>/dev/null || hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "output_dir=$output_dir"
df -h "$output_dir"

archive_path() {
    local object_name="$1"
    local structured="$output_dir/$object_name"
    local flattened="$output_dir/${object_name//\//_}"

    if [[ -f "$structured" ]]; then
        printf '%s\n' "$structured"
    else
        printf '%s\n' "$flattened"
    fi
}

if [[ "${TARTANAIR_SKIP_VALIDATION:-1}" != "1" ]]; then
    # Quarantine a partial/corrupt archive so a restarted job downloads it again.
    while read -r object_name; do
        archive=$(archive_path "$object_name")
        if [[ -f "$archive" ]] && ! unzip -tq "$archive" >/dev/null 2>&1; then
            mv -- "$archive" "$archive.corrupt.$SLURM_JOB_ID"
        fi
    done < <(awk '$1 ~ /(image|depth)_left[.]zip$/ {print $1}' download_training_zipfiles.txt)
fi

download_args=(
    --output-dir "$output_dir" \
    --rgb \
    --depth \
    --only-left \
    --unzip \
    --workers "${TARTANAIR_WORKERS:-8}"
)

if [[ "${TARTANAIR_SOURCE:-huggingface}" == "huggingface" ]]; then
    download_args+=(--huggingface)
fi

"$python_bin" -u download_training.py "${download_args[@]}"

if [[ "${TARTANAIR_SKIP_VALIDATION:-1}" == "1" ]]; then
    printf 'archives=72\nsequences=not_counted\nimages=not_counted\ndepths=not_counted\nvalidation=skipped\n' \
        > "$output_dir/TartanAir.ready"
    echo "TartanAir V1 download and extraction complete; validation skipped"
    cat "$output_dir/TartanAir.ready"
    exit 0
fi

archive_count=0
while read -r object_name; do
    archive=$(archive_path "$object_name")
    unzip -tq "$archive" >/dev/null
    archive_count=$((archive_count + 1))
done < <(awk '$1 ~ /(image|depth)_left[.]zip$/ {print $1}' download_training_zipfiles.txt)

sequence_count=$(find "$output_dir" -type f -name pose_left.txt | wc -l)
image_count=$(find "$output_dir" -type f -path '*/image_left/*.png' | wc -l)
depth_count=$(find "$output_dir" -type f -path '*/depth_left/*.npy' | wc -l)

if [[ "$archive_count" -ne 72 || "$sequence_count" -eq 0 || "$image_count" -eq 0 || "$depth_count" -eq 0 ]]; then
    echo "Dataset validation failed: archives=$archive_count sequences=$sequence_count images=$image_count depths=$depth_count" >&2
    exit 1
fi

printf 'archives=%s\nsequences=%s\nimages=%s\ndepths=%s\n' \
    "$archive_count" "$sequence_count" "$image_count" "$depth_count" \
    > "$output_dir/TartanAir.ready"

echo "TartanAir V1 download and extraction complete"
cat "$output_dir/TartanAir.ready"

#!/bin/bash -l
#SBATCH --job-name=dpvo-eval
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export LANG=C
export LC_ALL=C
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

module purge
module load "GCC/${DPVO_GCC_VERSION:-11.4.0}"
module load "CUDA/${DPVO_CUDA_VERSION:-12.1.1}"

project_dir="${DPVO_PROJECT_DIR:-/mnt/beegfs/xliu0001/DPVO}"
python_bin="${DPVO_PYTHON:-$project_dir/.conda-dpvo/bin/python}"
weights="${DPVO_EVAL_WEIGHTS:?Set DPVO_EVAL_WEIGHTS to a checkpoint path}"
split="${DPVO_EVAL_SPLIT:-validation}"
trials="${DPVO_EVAL_TRIALS:-1}"

cd "$project_dir"
mkdir -p slurm_logs TartanAirResults
test -f "$weights"
test -L datasets/TartanAir -o -d datasets/TartanAir

echo "timestamp=$(date --iso-8601=seconds)"
echo "host=$(hostname --fqdn 2>/dev/null || hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "weights=$weights"
echo "split=$split"
echo "trials=$trials"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

"$python_bin" -u evaluate_tartan.py \
    --weights "$weights" \
    --split "$split" \
    --trials "$trials"

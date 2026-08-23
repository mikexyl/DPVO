#!/bin/bash -l
#SBATCH --job-name=dpvo-v100
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=45-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export LANG=C
export LC_ALL=C
export TORCH_CUDA_ARCH_LIST=7.0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

module purge
module load "GCC/${DPVO_GCC_VERSION:-11.4.0}"
module load "CUDA/${DPVO_CUDA_VERSION:-12.1.1}"

project_dir="${DPVO_PROJECT_DIR:-/mnt/beegfs/xliu0001/DPVO}"
python_bin="${DPVO_PYTHON:-$project_dir/.conda-dpvo/bin/python}"
dataset="${DPVO_DATASET:-tartan}"
dataset_root="${DPVO_DATASET_ROOT:-/mnt/beegfs/xliu0001/datasets/TartanAir}"
run_name="${DPVO_RUN_NAME:-dpvo_tartanair_v1_1xv100_20260823}"
steps="${DPVO_STEPS:-240000}"
n_frames="${DPVO_N_FRAMES:-15}"

cd "$project_dir"
mkdir -p checkpoints runs slurm_logs

test -f "$project_dir/.conda-dpvo/DPVO_ENV_READY"
if [[ "$dataset" == "tartan" ]]; then
    test -f "$dataset_root/TartanAir.ready"
    test -f "$project_dir/datasets/TartanAir.pickle"
fi

echo "timestamp=$(date --iso-8601=seconds)"
echo "host=$(hostname --fqdn 2>/dev/null || hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
"$python_bin" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")'

train_args=(
    --dataset "$dataset"
    --datapath "$dataset_root"
    --name "$run_name"
    --steps "$steps"
    --n_frames "$n_frames"
    --num_workers 4
    --checkpoint_freq "${DPVO_CHECKPOINT_FREQ:-1000}"
    --lr 0.00008
)

if [[ -n "${DPVO_CKPT:-}" ]]; then
    train_args+=(--ckpt "$DPVO_CKPT")
fi

if [[ "${DPVO_SKIP_VALIDATION:-1}" == "1" ]]; then
    train_args+=(--skip_validation)
fi

"$python_bin" -u train.py "${train_args[@]}"

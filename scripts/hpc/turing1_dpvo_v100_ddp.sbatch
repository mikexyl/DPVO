#!/bin/bash -l
#SBATCH --job-name=dpvo-v100-ddp
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=192G
#SBATCH --time=20-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export LANG=C
export LC_ALL=C
export TORCH_CUDA_ARCH_LIST=7.0
export OMP_NUM_THREADS="${DPVO_OMP_THREADS:-6}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

module purge
module load "GCC/${DPVO_GCC_VERSION:-11.4.0}"
module load "CUDA/${DPVO_CUDA_VERSION:-12.1.1}"

project_dir="${DPVO_PROJECT_DIR:-/mnt/beegfs/xliu0001/DPVO}"
python_bin="${DPVO_PYTHON:-$project_dir/.conda-dpvo/bin/python}"
dataset="${DPVO_DATASET:-tartan}"
dataset_root="${DPVO_DATASET_ROOT:-/mnt/beegfs/xliu0001/datasets/TartanAir}"
run_name="${DPVO_RUN_NAME:-dpvo_tartanair_v1_3xv100_20260823}"
# Three ranks process the same sample budget as 240k single-GPU steps in 80k updates.
steps="${DPVO_STEPS:-80000}"
n_frames="${DPVO_N_FRAMES:-15}"
gpus_per_node="${DPVO_GPUS_PER_NODE:-3}"

cd "$project_dir"
mkdir -p checkpoints runs slurm_logs

test -f "$project_dir/.conda-dpvo/DPVO_ENV_READY"
if [[ "$dataset" == "tartan" ]]; then
    test -f "$dataset_root/TartanAir.ready"
    test -f "$project_dir/datasets/TartanAir.pickle"
fi

if [[ "${CUDA_VISIBLE_DEVICES//,/ }" != *" "* ]] && [[ "$gpus_per_node" -gt 1 ]]; then
    echo "Expected $gpus_per_node allocated GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi

echo "timestamp=$(date --iso-8601=seconds)"
echo "host=$(hostname --fqdn 2>/dev/null || hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "world_size=$gpus_per_node"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

train_args=(
    --dataset "$dataset"
    --datapath "$dataset_root"
    --name "$run_name"
    --steps "$steps"
    --n_frames "$n_frames"
    --num_workers "${DPVO_NUM_WORKERS:-4}"
    --checkpoint_freq "${DPVO_CHECKPOINT_FREQ:-1000}"
    --lr "${DPVO_LR:-0.00024}"
)

if [[ -n "${DPVO_CKPT:-}" ]]; then
    train_args+=(--ckpt "$DPVO_CKPT")
fi

if [[ "${DPVO_SKIP_VALIDATION:-1}" == "1" ]]; then
    train_args+=(--skip_validation)
fi

"$python_bin" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$gpus_per_node" \
    train.py "${train_args[@]}"

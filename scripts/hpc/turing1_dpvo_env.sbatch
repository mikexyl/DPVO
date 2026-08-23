#!/bin/bash -l
#SBATCH --job-name=dpvo-env
#SBATCH --partition=COMPUTE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export LANG=C
export LC_ALL=C
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export TORCH_CUDA_ARCH_LIST=7.0
export PIP_CACHE_DIR="${DPVO_PIP_CACHE:-/mnt/beegfs/xliu0001/.cache/pip}"
export TMPDIR="${DPVO_BUILD_TMP:-/mnt/beegfs/xliu0001/tmp/dpvo-env-${SLURM_JOB_ID}}"

module purge
module load Anaconda3/2024.06-1
module load GCC/11.4.0
module load CUDA/12.1.1

project_dir="${DPVO_PROJECT_DIR:-/mnt/beegfs/xliu0001/DPVO}"
env_dir="${DPVO_ENV_DIR:-$project_dir/.conda-dpvo}"

mkdir -p "$project_dir/slurm_logs" "$PIP_CACHE_DIR" "$TMPDIR"
cd "$project_dir"

echo "timestamp=$(date --iso-8601=seconds)"
echo "host=$(hostname --fqdn 2>/dev/null || hostname)"
echo "env_dir=$env_dir"
gcc --version | head -n 1
nvcc --version | tail -n 1

if [[ ! -x "$env_dir/bin/python" ]]; then
    conda create --yes --prefix "$env_dir" python=3.10 pip
fi

"$env_dir/bin/pip" install --upgrade pip setuptools wheel ninja
"$env_dir/bin/pip" install --retries 30 --resume-retries 30 --timeout 3600 \
    numpy==1.26.4 pillow==10.4.0 pybind11
"$env_dir/bin/pip" install --retries 30 --resume-retries 30 --timeout 3600 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
"$env_dir/bin/pip" install \
    torch-scatter==2.1.2 \
    --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html
"$env_dir/bin/pip" install \
    tensorboard numba tqdm einops pypose kornia numpy==1.26.4 \
    plyfile evo opencv-python yacs matplotlib

"$env_dir/bin/pip" install --no-build-isolation --force-reinstall --no-deps .

"$env_dir/bin/python" - <<'PY'
import torch
import cuda_ba
import cuda_corr
import lietorch_backends

assert torch.__version__.startswith("2.3.1"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print("DPVO extension imports: OK")
PY

touch "$env_dir/DPVO_ENV_READY"
echo "DPVO environment build complete"

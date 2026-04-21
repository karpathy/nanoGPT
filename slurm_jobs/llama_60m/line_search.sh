#!/bin/bash
#SBATCH --job-name=ls_llama60m
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,apollo_agate
#SBATCH --chdir=/users/9/chen8596/nanoGPT
#SBATCH --output=/users/9/chen8596/nanoGPT/exp_log/llama60m_linesearch_%A_%a.out
#SBATCH --error=/users/9/chen8596/nanoGPT/exp_log/llama60m_linesearch_%A_%a.err


set -euo pipefail

REPO_ROOT="/users/9/chen8596/nanoGPT"
CONDA_SH="/users/9/chen8596/miniconda3/etc/profile.d/conda.sh"

mkdir -p "${REPO_ROOT}/exp_log/slurm"
cd "${REPO_ROOT}"

if [ ! -f "${CONDA_SH}" ]; then
  echo "Missing Conda init script at ${CONDA_SH}" >&2
  exit 127
fi
source "${CONDA_SH}"
conda activate nanogpt
echo "Python: $(command -v python)"
python --version

MAX_ITERS="${MAX_ITERS:-3600}"
RUN_ROOT="/scratch.global/chen8596/experiment_runs/llama60m_line_search_stage2_maxiters_${MAX_ITERS}"

python run_linesearch_stage2.py \
  --run-root "$RUN_ROOT" \
  --train-script "train_linesearch_llama.py" \
  --config-path "config/train_llama_60m.py" \
  --nproc-per-node 4 \
  --experiment-name "llama60m_line_search" \
  --trial-id "stage2_final" \
  -- config/train_llama_60m.py --max_iters="$MAX_ITERS" --lr_decay_iters="$MAX_ITERS"

#!/bin/bash
#SBATCH --job-name=sens_ls_adam_gpt2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,apollo_agate
#SBATCH --chdir=/users/9/chen8596/nanoGPT
#SBATCH --array=0-9
#SBATCH --output=/users/9/chen8596/nanoGPT/exp_log/gpt2_sens_ls_adam_%A_%a.out
#SBATCH --error=/users/9/chen8596/nanoGPT/exp_log/gpt2_sens_ls_adam_%A_%a.err


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

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
echo "Array task ${TASK_ID} -> sweep=linesearch_adam_c1 trial_index=${TASK_ID}"

python run_sensitivity.py \
  config/experiments/sensitivity_analysis_gpt.yaml \
  --only linesearch_adam_c1 \
  --trial-index "${TASK_ID}" \
  --resume

#!/bin/bash
#SBATCH --job-name=n12_llama60_sfa_nc
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,apollo_agate
#SBATCH --chdir=/users/9/chen8596/nanoGPT
#SBATCH --output=/users/9/chen8596/nanoGPT/exp_log/llama60m_sfa_nc_n12_%A_%a.out
#SBATCH --error=/users/9/chen8596/nanoGPT/exp_log/llama60m_sfa_nc_n12_%A_%a.err


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
RUN_ROOT="/scratch.global/chen8596/experiment_runs/llama60m_schedulefree_adam_noclip_lr_search_serial_halving_num_trials_12_maxiters_${MAX_ITERS}"
echo "Launching or resuming serial halving run at $RUN_ROOT"

python run_stage1_optuna.py config/experiments/optuna_schedulefree_adam_llama60m_noclip.yaml \
  --num-trials 12 \
  --num-iterations-per-trial "$MAX_ITERS" \
  --run-root "$RUN_ROOT"

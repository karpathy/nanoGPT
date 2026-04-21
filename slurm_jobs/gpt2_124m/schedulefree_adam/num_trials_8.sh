#!/bin/bash
#SBATCH --job-name=n8_gpt124_sfa
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=jinzn
#SBATCH --gres=gpu:4
#SBATCH -p a100-4,apollo_agate
#SBATCH --chdir=/users/9/chen8596/nanoGPT
#SBATCH --output=/users/9/chen8596/nanoGPT/exp_log/gpt124m_sfa_n8_%A_%a.out
#SBATCH --error=/users/9/chen8596/nanoGPT/exp_log/gpt124m_sfa_n8_%A_%a.err


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

MAX_ITERS="${MAX_ITERS:-5200}"
RUN_ROOT="/scratch.global/chen8596/experiment_runs/gpt124m_schedulefree_adam_lr_search_serial_halving_num_trials_8_maxiters_${MAX_ITERS}"
echo "Launching or resuming serial halving run at $RUN_ROOT"

python run_stage1_optuna.py config/experiments/optuna_schedulefree_adam_gpt124m.yaml \
  --num-trials 8 \
  --reduction-factor 4 \
  --num-iterations-per-trial "$MAX_ITERS" \
  --run-root "$RUN_ROOT"

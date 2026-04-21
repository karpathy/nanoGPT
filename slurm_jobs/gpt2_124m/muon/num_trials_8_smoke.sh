#!/bin/bash
#SBATCH --job-name=n8_gpt124_muon_smk
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=bgop-delta-gpu
#SBATCH --gres=gpu:4
#SBATCH -p gpuA100x4
#SBATCH --output=exp_log/gpt124m_muon_n8_smoke_%A_%a.out
#SBATCH --error=exp_log/gpt124m_muon_n8_smoke_%A_%a.err

set -euo pipefail

mkdir -p exp_log/slurm
cd ../../..

SMOKE_EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL:-2}"
mkdir -p slurm_jobs/gpt2_124m/smoke_configs/generated
SMOKE_CONFIG="slurm_jobs/gpt2_124m/smoke_configs/generated/optuna_muon_gpt124m_smoke_ei${SMOKE_EVAL_INTERVAL}.yaml"
cp slurm_jobs/gpt2_124m/smoke_configs/optuna_muon_gpt124m_smoke.yaml "$SMOKE_CONFIG"
sed -i "s/__SMOKE_EVAL_INTERVAL__/${SMOKE_EVAL_INTERVAL}/g" "$SMOKE_CONFIG"

RUN_ROOT="/work/nvme/bgop/cchen47/experiment_runs/gpt124m_muon_lr_search_serial_halving_num_trials_8_smoke_ei${SMOKE_EVAL_INTERVAL}"
echo "Launching or resuming smoke serial halving run at $RUN_ROOT"

python run_stage1_optuna.py "$SMOKE_CONFIG" \
  --num-trials 8 \
  --reduction-factor 4 \
  --run-root "$RUN_ROOT"

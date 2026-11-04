#!/bin/bash
# One-GPU Slurm job. Edit the job name and the command at the bottom, then from this folder:
#   cd ~/workspace/bci_nanoGPT/slurm && sbatch submit.sh
# Logs: slurm/logs/<job name>-<job id>.log      Check: squeue -u jonas      Stop: scancel <job id>
#SBATCH --job-name=eeg-multi-prep
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/groups/bci/jonas/workspace/bci_nanoGPT/slurm/logs/%x-%j.log

set -euo pipefail
cd /data/groups/bci/jonas/workspace/bci_nanoGPT          # run from the repo top level, wherever you submitted from
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES"

# ---- the command (edit here; Slurm sets CUDA_VISIBLE_DEVICES) ----
/data/groups/bci/jonas/venv_dac_audio/bin/python -u data/eeg_single/prepare.py --n_train 300 --n_val 30 --out_dir data/eeg_multi
# /data/groups/bci/jonas/venv_dac/bin/python -u train.py config/train_eeg_multi.py --out_dir=out/eeg-multi-none --wandb_run_name=eeg-multi-none

echo "[$(date '+%F %T')] done"

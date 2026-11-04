#!/bin/bash
# One-GPU Slurm job, or several in parallel as a job array (one GPU each).
#   cd ~/workspace/bci_nanoGPT/slurm && sbatch submit.sh
# Edit the job name, the --array line and the command at the bottom.
# Logs: slurm/logs/<job name>-<job id>_<array index>.log      Check: squeue -u jonas      Stop: scancel <job id>
#SBATCH --job-name=eeg-multi-enc
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/groups/bci/jonas/workspace/bci_nanoGPT/slurm/logs/%x-%A_%a.log

set -euo pipefail
cd /data/groups/bci/jonas/workspace/bci_nanoGPT          # run from the repo top level, wherever you submitted from
echo "[$(date '+%F %T')] job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), GPU $CUDA_VISIBLE_DEVICES"

# ---- the command (edit here; Slurm sets CUDA_VISIBLE_DEVICES) ----
# job array: task i runs the i-th entry (--array=0-1 above = 2 parallel jobs; add entries and widen --array for more)
ENCODINGS=(none learned)                                  # later: (none learned sinusoidal rope4d) with --array=0-3
enc=${ENCODINGS[${SLURM_ARRAY_TASK_ID:-0}]}
/data/groups/bci/jonas/venv_dac/bin/python -u train.py config/train_eeg_multi.py --channel_encoding=$enc \
    --out_dir=out/eeg-multi-$enc --wandb_run_name=eeg-multi-$enc

# single job instead: delete the --array line above and use one plain command, e.g. the token preprocessing:
# /data/groups/bci/jonas/venv_dac_audio/bin/python -u data/eeg_single/prepare.py --n_train 300 --n_val 30 --out_dir data/eeg_multi

echo "[$(date '+%F %T')] done"

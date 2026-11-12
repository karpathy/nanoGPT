#!/bin/bash
# One-GPU Slurm job per run; several runs in parallel as a job array (one GPU each).
#   cd ~/workspace/bci_nanoGPT/slurm && sbatch submit.sh
# For a new round: edit TAG, NAMES, ARGS below and the --array range (0 .. number of runs - 1).
# Logs: slurm/logs/<job name>-<job id>_<array index>.log      Check: squeue -u jonas      Stop: scancel <job id>
#SBATCH --job-name=eeg-classify
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/groups/bci/jonas/workspace/bci_nanoGPT/slurm/logs/%x-%A_%a.log

set -euo pipefail
cd /data/groups/bci/jonas/workspace/bci_nanoGPT          # run from the repo top level, wherever you submitted from
echo "[$(date '+%F %T')] job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), GPU $CUDA_VISIBLE_DEVICES"

# ---- the runs of this round (edit here) ----
# Everything not listed in ARGS comes from CONFIG. Task i runs NAMES[i] with ARGS[i]; wandb run = <PREFIX>-<name>-<TAG>.
# {params} in a name becomes the parameter count in the wandb name (e.g. 13.8M); out_dir leaves it out.
TAG=r5
# Round 5: classifier on Hubner2016 P3 (1 s windows = 8 frames, Target vs NonTarget ~1:3), 3k steps, lr 1e-4, full attention.
# The three ways to use pretraining, plus labels only. MLM = round-3 masked RoPE d384-l6 (20k steps), NTP = round-4 next-token RoPE d384-l6.
CONFIG=config/train_eeg_classify.py
PREFIX=eeg-classify
MLM=out/eeg-multi-rope-d384-l6-r3/ckpt.pt
NTP=out/eeg-multi-ntp-rope-d384-l6-r4/ckpt.pt
NAMES=(hubner-mlm-probe                                 hubner-mlm-finetune            hubner-ntp-finetune            hubner-scratch)
ARGS=("--pretrained_ckpt=$MLM --freeze_backbone=True"   "--pretrained_ckpt=$MLM"       "--pretrained_ckpt=$NTP"       "")
# Round 4 (done, job 88360): TAG=r4, CONFIG=config/train_eeg_multi.py, PREFIX=eeg-multi, next-token (frame-causal), 5k steps:
#   N="--bidirectional=False"; R="$N --channel_encoding=rope4d"
#   NAMES=(ntp-none-{params}-d384-l6 ntp-learned-{params}-d384-l6 ntp-sinusoidal-{params}-d384-l6 ntp-rope-{params}-d384-l6 ntp-rope-{params}-d256-l4 ntp-rope-{params}-d512-l8 ntp-rope-{params}-d384-l12)
#   ARGS=("$N --channel_encoding=none" "$N --channel_encoding=learned" "$N --channel_encoding=sinusoidal" "$R" "$R --n_embd=256 --n_head=4 --n_layer=4" "$R --n_embd=512 --n_head=8 --n_layer=8" "$R --n_embd=384 --n_head=6 --n_layer=12")
# Round 3 (done, job 87842): TAG=r3, masked, RoPE, 20k steps: NAMES=(rope-d256-l4 rope-d384-l6 rope-d512-l8 rope-d384-l12)
#   L="--channel_encoding=rope4d --max_iters=20000 --lr_decay_iters=20000"
#   ARGS=("$L --n_embd=256 --n_head=4 --n_layer=4" "$L --n_embd=384 --n_head=6 --n_layer=6" "$L --n_embd=512 --n_head=8 --n_layer=8" "$L --n_embd=384 --n_head=6 --n_layer=12")
# Round 2 (done, job 87818): TAG=r2  NAMES=(lr3e-4 lr3e-3 clip5)  ARGS: learned with --learning_rate=3e-4 --min_lr=3e-5 | --learning_rate=3e-3 --min_lr=3e-4 | --grad_clip=5.0
# Round 1 (done, job 87816): TAG=r1  NAMES=(rope4d learned-seed2)  ARGS=("--channel_encoding=rope4d" "--channel_encoding=learned --seed=2")

# Later rounds (copy over the three lines above, set --array to match):
# Round 2, learning rate (on the Round-1 winner, here learned), --array=0-2:
#   TAG=r2
#   NAMES=(lr3e-4                                                   lr3e-3                                                   clip5)
#   ARGS=("--channel_encoding=learned --learning_rate=3e-4 --min_lr=3e-5"  "--channel_encoding=learned --learning_rate=3e-3 --min_lr=3e-4"  "--channel_encoding=learned --grad_clip=5.0")
# Round 3, training length, --array=0-0:
#   TAG=r3
#   NAMES=(20k)
#   ARGS=("--channel_encoding=learned --max_iters=20000 --lr_decay_iters=20000")
# Round 4, model size at 20k steps, --array=0-1:
#   TAG=r4
#   NAMES=(dim256                                                              dim512-l8)
#   ARGS=("--channel_encoding=learned --max_iters=20000 --lr_decay_iters=20000 --n_embd=256"  "--channel_encoding=learned --max_iters=20000 --lr_decay_iters=20000 --n_embd=512 --n_layer=8")

i=${SLURM_ARRAY_TASK_ID:-0}
/data/groups/bci/jonas/venv_dac/bin/python -u train.py $CONFIG ${ARGS[$i]} \
    --out_dir=out/$PREFIX-${NAMES[$i]//\{params\}-/}-$TAG --wandb_run_name=$PREFIX-${NAMES[$i]}-$TAG

# single job instead (e.g. the token preprocessing): delete the --array line and use one plain command:
# /data/groups/bci/jonas/venv_dac_audio/bin/python -u data/eeg_single/prepare.py --n_train 300 --n_val 30 --out_dir data/eeg_multi

echo "[$(date '+%F %T')] done"

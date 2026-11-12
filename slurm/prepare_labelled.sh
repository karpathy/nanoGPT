#!/bin/bash
# Tokenize the labelled NeuralBench pilot sets for the classifier, one GPU per dataset.
#   cd ~/workspace/bci_nanoGPT/slurm && sbatch prepare_labelled.sh
# Tokens go to /data/groups/bci/datasets/processed/dac_tokens/<DAC run>/<Study>_<task>/ (a derived dataset, not repo
# data); train.py reads data/<name>, so data/eeg_<name> is a symlink to that folder.
#SBATCH --job-name=eeg-labelled-prep
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/groups/bci/jonas/workspace/bci_nanoGPT/slurm/logs/%x-%A_%a.log

set -euo pipefail
cd /data/groups/bci/jonas/workspace/bci_nanoGPT
echo "[$(date '+%F %T')] job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), GPU $CUDA_VISIBLE_DEVICES"

PILOT=/data/groups/bci/datasets/processed/v8.1_neuralbench_pilot/tasks
TOKENS=/data/groups/bci/datasets/processed/dac_tokens/rand10k_baseline       # the DAC run of data/eeg_single/prepare.py's CKPT
CACHES=($PILOT/p3/Korczowski2019BrainBi2014B/cache.json  $PILOT/p3/Hubner2016Eeg/cache.json  $PILOT/p3/Schreuder2010New/cache.json  $PILOT/cvep/CabreraCastillos2023BurstCvep40/cache.json)
NAMES=(Korczowski2019BrainBi2014B_p3                      Hubner2016Eeg_p3                     Schreuder2010New_p3                     CabreraCastillos2023BurstCvep40_cvep)
# Hamid2020Tuar (artifact) is multilabel: not supported by prepare.py yet.

i=${SLURM_ARRAY_TASK_ID:-0}
NUMPY_MADVISE_HUGEPAGE=0 /data/groups/bci/jonas/venv_dac_audio/bin/python -u data/eeg_labelled/prepare.py \
    --cache ${CACHES[$i]} --out_dir $TOKENS/${NAMES[$i]}

echo "[$(date '+%F %T')] done"

#!/bin/bash
# One-GPU Slurm job, or several in parallel as a job array (one GPU each).
#   cd ~/workspace/bci_nanoGPT/slurm && sbatch submit_cw.sh
# Training runs come from slurm/run_list.txt (one run per line: dataset encoding seed); array task N runs the N-th
# non-comment line, so set --array=1-<number of runs>. Rerun failures with e.g. --array=5,11.
# Logs: slurm/logs/<job name>-<job id>_<array index>.log      Check: squeue -u chris      Stop: scancel <job id>
#SBATCH --job-name=eeg-multi-enc
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=1-4                                                # = number of runs in run_list.txt
#SBATCH --output=/data/groups/bci/chris/workspace/bci_nanoGPT/slurm/logs/%x-%A_%a.log

set -euo pipefail
cd /data/groups/bci/chris/workspace/bci_nanoGPT          # run from the repo top level, wherever you submitted from
echo "[$(date '+%F %T')] job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), GPU $CUDA_VISIBLE_DEVICES"


# (0). Regardless of whether 1 or 2, run this block.
data_base_dir=/data/groups/bci/datasets/processed/dac_tokens/eeg_multi_cw

#
# # # # # # # # # # # # # # # # # # # # # # 
#

# (1). Train the nanoGPT transformer: one run per array task, read from slurm/run_list.txt ----

# TAG=metrics                                                         # suffix: new run names and folders, old runs stay untouched
RUN_LIST=slurm/run_list.txt
read -r ds enc seed < <(grep -v '^#\|^$' $RUN_LIST | sed -n "${SLURM_ARRAY_TASK_ID:-1}p")
[ -n "${seed:-}" ] || { echo "no line ${SLURM_ARRAY_TASK_ID:-1} in $RUN_LIST"; exit 1; }
name=$ds-$enc-s$seed #-$TAG                               # e.g. RD_hop_006-learned-s0
echo "dataset $ds   encoding $enc   seed $seed   run $name"
/data/groups/bci/chris/workspace/venv_dac/bin/python -u train_cw.py config/train_eeg_multi_cw.py \
    --dataset=$data_base_dir/$ds \
    --channel_encoding=$enc \
    --seed=$seed \
    --out_dir=out/$name \
    --wandb_run_name=$name
    # Note: --compile=False was once needed because of a torch.compile error (CW)
    # Upgrading to torch 2.9.1 and triton 3.5.1 allows us to use torch.compile (CW)

#
# # # # # # # # # # # # # # # # # # # # # # 
#

# (2). To run the token preprocessing to prepare data for training nanoGPT model: 
#      Is frozen dataset and samples are comparable across checkpoints and different tokenizations if n_train and n_val unchanged and if 
#      /data/groups/bci/datasets/processed/v8_sets/10k is unchanged.
#
# # EXAMPLE: /data/groups/bci/chris/workspace/venv_dac/bin/python -u data/eeg_single/prepare_cw.py --n_train 300 --n_val 30 --out_dir data/eeg_multi

# datasets=(vq_long_006 RD_hop_000 RD_hop_003 RD_hop_006)                               
# ds=${datasets[${SLURM_ARRAY_TASK_ID:-0}]}
# /data/groups/bci/chris/workspace/venv_dac/bin/python -u data/eeg_single/prepare_cw.py --n_train 300 --n_val 30 --out_dir $data_base_dir/$ds --ckpt $ds


#
# # # # # # # # # # # # # # # # # # # # # # 
#

echo "[$(date '+%F %T')] done"

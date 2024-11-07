#!/bin/bash

# Parameters
#SBATCH --account=compute-account
#SBATCH --dependency=singleton
#SBATCH --error=/srv/scratch/nanogpt/nanoGPT/results/nanotGPT_%j.err
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nanoGPT
#SBATCH --mem=0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=/srv/scratch/nanogpt/nanoGPT/results/nanotGPT_%j.out
#SBATCH --partition=batch
#SBATCH --time=0-01:00:00

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0

# Get the IP address of the first node
MASTER_ADDR=$(srun --ntasks=1 --nodes=1 hostname -I | awk '{print $1}')
MASTER_PORT=$(shuf -i 10000-65500 -n 1)

echo "Using MASTER_ADDR: $MASTER_ADDR"
echo "Using MASTER_PORT: $MASTER_PORT"

# command 1
srun --output /srv/scratch/nanogpt/nanoGPT/results/nanotGPT_%j_%t.out \
     --error /srv/scratch/nanogpt/nanoGPT/results/nanotGPT_%j_%t.err \
     --container-image /srv/scratch/nanogpt/nanoGPT/nvidia+pytorch+24.08-py3.sqsh \
     --container-mounts /srv/scratch/nanogpt/nanoGPT:/workspace \
     --no-container-mount-home \
     bash -c "CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
       --nnodes=$SLURM_NNODES \
       --nproc_per_node=$SLURM_GPUS_PER_NODE \
       --node_rank=\$SLURM_PROCID \
       --master_addr=$MASTER_ADDR \
       --master_port=$MASTER_PORT \
       train.py config/train_gpt2.py"
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_xl.py
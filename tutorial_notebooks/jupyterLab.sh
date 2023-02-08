#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition mldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=logs/jupyter.log


cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888

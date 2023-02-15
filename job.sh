#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080 #rtx2080
#SBATCH --gres=gpu:1
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J char_llm # sets the job name. If not specified, the file name will be used as job name
python archs_playground/character_llm.py


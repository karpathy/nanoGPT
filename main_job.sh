#!/bin/bash 
#SBATCH -p general ## Partition
#SBATCH -q public  ## QOS
#SBATCH -N 1      ## Number of Sol Nodes
#SBATCH -c 8     ## Number of Cores
#SBATCH --mem=32G  ## Memory (GB)
#SBATCH --time=10  ## Minutes of compute
#SBATCH -G 1        ## Number of GPUs
#SBATCH --job-name=large-caissa-test
#SBATCH --output=slurm.%j.out  ## job /dev/stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   ## job /dev/stderr record 
#SBATCH --export=NONE          ## keep environment clean
#SBATCH --mail-type=ALL        ## notify <asurite>@asu.edu for any job state change

echo "=========================================="
echo "Large Caissa Test"
echo "=========================================="

# Load environment
module load mamba/latest

# Install dependencies
echo "Installing mamba stuffs"
if ! mamba env list | grep -q "caissa-env"; then
    echo "Creating new mamba environment caissa-env"
    mamba create -y -n caissa-env python=3.10
fi

# Activate env and install dependencies
source activate caissa-env
echo "Installing Python stuffs"
pip install torch numpy transformers datasets tiktoken wandb tqdm

# Check if model already exists
if [ -f "output/ckpt.pt" ]; then
    echo "=========================================="
    echo "Found existing trained model, skipping training"
    echo "=========================================="
else
    echo "=========================================="
    echo "No existing model found, starting training"
    echo "=========================================="

    # Prepare training data (only if not already prepared)
    if [ -f "data/chess-data/train.bin" ] && [ -f "data/chess-data/val.bin" ]; then
        echo "Training data already exists, skipping preparation"
    else
        echo "Preparing training data"
        python data/chess-data/prepare.py
    fi

    # Train LLM
    echo "Training LLM on data"
    python train.py config/train_caissa.py
fi

# Sample from LLM
echo "=========================================="
echo "Sampling from trained model"
echo "=========================================="
echo "Test sample LLM without prompt"
python sample.py --out_dir=output

echo "Test sample LLM with prompt"
python sample.py --start="FILE:./prompts/prompt.txt" --out_dir=output
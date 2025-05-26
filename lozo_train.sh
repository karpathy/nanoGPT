#!/bin/bash
# This script runs LOZO training for GPT-2 pretraining
# Usage: bash lozo_train.sh [lozo|lozom|svdlozo|mezo|mezom|dimezo|dilozo|adam|sgd] [single|multi] [dataset] [adaptive|rank-adaptive]

METHOD=${1:-lozo}      # lozo, lozom, svdlozo, mezo, mezom, dimezo, dilozo, adam, or sgd
MODE=${2:-single}      # single or multi (for multi-GPU)
DATASET=${3:-openwebtext}  # dataset name: openwebtext, shakespeare, etc.
ADAPTIVE=${4:-}        # optional: "adaptive" for adaptive zo_eps variants, "rank-adaptive" for rank scheduling

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Select config file based on method and dataset
if [ "$DATASET" = "shakespeare" ]; then
    if [ "$METHOD" = "lozo" ]; then
        if [ "$ADAPTIVE" = "rank-adaptive" ]; then
            CONFIG="config/train_rank_adaptive_lozo_shakespeare.py"
            echo "Running Rank-Adaptive LOZO training on Shakespeare with config: $CONFIG"
        else
            CONFIG="config/train_lozo_shakespeare.py"
            echo "Running standard LOZO training on Shakespeare with config: $CONFIG"
        fi
    elif [ "$METHOD" = "lozom" ]; then
        CONFIG="config/train_lozom_shakespeare.py"
        echo "Running LOZO-M (with momentum) training on Shakespeare with config: $CONFIG"
    elif [ "$METHOD" = "svdlozo" ]; then
        CONFIG="config/train_svdlozo_shakespeare.py"
        echo "Running SVD-LOZO training on Shakespeare with config: $CONFIG"
    elif [ "$METHOD" = "mezo" ]; then
        CONFIG="config/train_mezo_shakespeare.py"
        echo "Running MeZO training on Shakespeare with config: $CONFIG"
    elif [ "$METHOD" = "mezom" ]; then
        CONFIG="config/train_mezom_shakespeare.py"
        echo "Running MeZO-M (with momentum) training on Shakespeare with config: $CONFIG"
    elif [ "$METHOD" = "dimezo" ]; then
        if [ "$ADAPTIVE" = "adaptive" ]; then
            CONFIG="config/train_dimezo_adaptive_shakespeare.py"
            echo "Running DiMeZO with Adaptive zo_eps training on Shakespeare with config: $CONFIG"
        else
            CONFIG="config/train_dimezo_shakespeare.py"
            echo "Running DiMeZO training on Shakespeare with config: $CONFIG"
        fi
    elif [ "$METHOD" = "dilozo" ]; then
        if [ "$ADAPTIVE" = "adaptive" ]; then
            CONFIG="config/train_dilozo_adaptive_shakespeare.py"
            echo "Running DiLoZO with Adaptive zo_eps training on Shakespeare with config: $CONFIG"
        elif [ "$ADAPTIVE" = "rank-adaptive" ]; then
            CONFIG="config/train_dilozo_rank_adaptive_shakespeare.py"
            echo "Running DiLoZO with Rank Adaptive training on Shakespeare with config: $CONFIG"
        else
            CONFIG="config/train_dilozo_shakespeare.py"
            echo "Running DiLoZO training on Shakespeare with config: $CONFIG"
        fi
    elif [ "$METHOD" = "adam" ]; then
        CONFIG="config/train_adam_shakespeare.py"
        echo "Running Adam optimizer training on Shakespeare with config: $CONFIG"
    elif [ "$METHOD" = "sgd" ]; then
        CONFIG="config/train_sgd_shakespeare.py"
        echo "Running SGD optimizer training on Shakespeare with config: $CONFIG"
    else
        echo "Invalid method. Use 'lozo', 'lozom', 'svdlozo', 'mezo', 'mezom', 'dimezo', 'dilozo', 'adam', or 'sgd'"
        exit 1
    fi
else
    # Default GPT-2 configs for other datasets
    if [ "$METHOD" = "lozo" ]; then
        if [ "$ADAPTIVE" = "rank-adaptive" ]; then
            CONFIG="config/train_rank_adaptive_lozo_gpt2.py"
            echo "Running Rank-Adaptive LOZO training with config: $CONFIG"
        else
            CONFIG="config/train_lozo_gpt2.py"
            echo "Running standard LOZO training with config: $CONFIG"
        fi
    elif [ "$METHOD" = "lozom" ]; then
        CONFIG="config/train_lozom_gpt2.py"
        echo "Running LOZO-M (with momentum) training with config: $CONFIG"
    elif [ "$METHOD" = "svdlozo" ]; then
        CONFIG="config/train_svdlozo_gpt2.py"
        echo "Running SVD-LOZO training with config: $CONFIG"
    elif [ "$METHOD" = "mezo" ]; then
        CONFIG="config/train_mezo_gpt2.py"
        echo "Running MeZO training with config: $CONFIG"
    elif [ "$METHOD" = "mezom" ]; then
        CONFIG="config/train_mezom_gpt2.py"
        echo "Running MeZO-M (with momentum) training with config: $CONFIG"
    elif [ "$METHOD" = "dimezo" ]; then
        if [ "$ADAPTIVE" = "adaptive" ]; then
            CONFIG="config/train_dimezo_adaptive_gpt2.py"
            echo "Running DiMeZO with Adaptive zo_eps training with config: $CONFIG"
        else
            CONFIG="config/train_dimezo_gpt2.py"
            echo "Running DiMeZO training with config: $CONFIG"
        fi
    elif [ "$METHOD" = "dilozo" ]; then
        if [ "$ADAPTIVE" = "adaptive" ]; then
            CONFIG="config/train_dilozo_adaptive_gpt2.py"
            echo "Running DiLoZO with Adaptive zo_eps training with config: $CONFIG"
        elif [ "$ADAPTIVE" = "rank-adaptive" ]; then
            CONFIG="config/train_dilozo_rank_adaptive_gpt2.py"
            echo "Running DiLoZO with Rank Adaptive training with config: $CONFIG"
        else
            CONFIG="config/train_dilozo_gpt2.py"
            echo "Running DiLoZO training with config: $CONFIG"
        fi
    elif [ "$METHOD" = "adam" ]; then
        CONFIG="config/train_adam_gpt2.py"
        echo "Running Adam optimizer training with config: $CONFIG"
    elif [ "$METHOD" = "sgd" ]; then
        CONFIG="config/train_sgd_gpt2.py"
        echo "Running SGD optimizer training with config: $CONFIG"
    else
        echo "Invalid method. Use 'lozo', 'lozom', 'svdlozo', 'mezo', 'mezom', 'dimezo', 'dilozo', 'adam', or 'sgd'"
        exit 1
    fi
fi

# Create log directory
LOG_DIR="logs/${METHOD}-${DATASET}"
mkdir -p "$LOG_DIR"

# Create unique log file name with timestamp
LOG_FILE="${LOG_DIR}/${METHOD}-${DATASET}-${TIMESTAMP}.log"

echo "Logs will be stored in: $LOG_FILE"
echo "Using config file: $CONFIG"

# Set up the training command
if [ "$MODE" = "single" ]; then
    # Single GPU training
    echo "Running on single GPU"
    
    # Use the appropriate training script based on method
    if [ "$METHOD" = "adam" ] || [ "$METHOD" = "sgd" ]; then
        # For Adam or SGD, use the standard train.py script
        echo "Using standard PyTorch optimization with train.py"
        python train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "lozo" ] || [ "$METHOD" = "lozom" ]; then
        # For LOZO methods, use lozo_train.py
        echo "Using LOZO optimization with lozo_train.py"
        python lozo_train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "svdlozo" ]; then
        # For SVD-LOZO method, use lozo_train.py
        echo "Using SVD-LOZO optimization with lozo_train.py"
        python lozo_train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "mezo" ] || [ "$METHOD" = "mezom" ]; then
        # For MeZO methods, use mezo_train.py
        echo "Using MeZO optimization with mezo_train.py"
        python mezo_train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "dimezo" ]; then
        # For DiMeZO method, use lozo_train.py
        echo "Using DiMeZO optimization with lozo_train.py"
        python lozo_train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "dilozo" ]; then
        # For DiLoZO method, use lozo_train.py
        echo "Using DiLoZO optimization with lozo_train.py"
        python lozo_train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    else
        echo "Invalid method. Use 'lozo', 'lozom', 'svdlozo', 'mezo', 'mezom', 'dimezo', 'dilozo', 'adam', or 'sgd'"
        exit 1
    fi
elif [ "$MODE" = "multi" ]; then
    # Multi-GPU training with DDP
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Running on $NUM_GPUS GPUs"
    
    # Use the appropriate training script based on method
    if [ "$METHOD" = "adam" ] || [ "$METHOD" = "sgd" ]; then
        # For Adam or SGD, use the standard train.py script
        echo "Using standard PyTorch optimization with train.py"
        torchrun --standalone --nproc_per_node=$NUM_GPUS train.py $CONFIG --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "lozo" ] || [ "$METHOD" = "lozom" ]; then
        # For LOZO methods, use lozo_train.py
        echo "Using LOZO optimization with lozo_train.py"
        torchrun --standalone --nproc_per_node=$NUM_GPUS lozo_train.py $CONFIG --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "svdlozo" ]; then
        # For SVD-LOZO method, use lozo_train.py
        echo "Using SVD-LOZO optimization with lozo_train.py"
        torchrun --standalone --nproc_per_node=$NUM_GPUS lozo_train.py $CONFIG --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "mezo" ] || [ "$METHOD" = "mezom" ]; then
        # For MeZO methods, use mezo_train.py
        echo "Using MeZO optimization with mezo_train.py"
        torchrun --standalone --nproc_per_node=$NUM_GPUS mezo_train.py $CONFIG --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "dimezo" ]; then
        # For DiMeZO method, use lozo_train.py
        echo "Using DiMeZO optimization with lozo_train.py"
        torchrun --standalone --nproc_per_node=$NUM_GPUS lozo_train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    elif [ "$METHOD" = "dilozo" ]; then
        # For DiLoZO method, use lozo_train.py
        echo "Using DiLoZO optimization with lozo_train.py"
        torchrun --standalone --nproc_per_node=$NUM_GPUS lozo_train.py $CONFIG --batch_size=32 --compile=True --dataset=$DATASET 2>&1 | tee "$LOG_FILE"
    else
        echo "Invalid method. Use 'lozo', 'lozom', 'svdlozo', 'mezo', 'mezom', 'dimezo', 'dilozo', 'adam', or 'sgd'"
        exit 1
    fi
else
    echo "Invalid mode. Use 'single' or 'multi'"
    exit 1
fi 
# Configuration for GPT-2 training with SVD enabled
# This config demonstrates how to use SVD with different settings

wandb_log = False  # Disable wandb for quick testing
wandb_project = 'svd-experiments'
wandb_run_name = 'gpt2-svd-test'

# Smaller scale for testing SVD functionality
batch_size = 4
block_size = 256
gradient_accumulation_steps = 1

# Shorter training for testing
max_iters = 1000
lr_decay_iters = 1000

# More frequent evaluation for monitoring SVD impact
eval_interval = 100
eval_iters = 50
log_interval = 10

# Standard hyperparameters
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Model configuration
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# SVD Configuration - ENABLED for testing
use_svd = True        # Enable SVD on value matrices
svd_rank = 16         # Use rank-16 approximation

# Dataset
dataset = 'shakespeare_char'
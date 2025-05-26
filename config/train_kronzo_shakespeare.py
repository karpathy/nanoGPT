# Config for KronZO training on Shakespeare dataset

import torch

# Output directory
out_dir = 'out-kronzo-shakespeare'

# Logging settings
eval_interval = 500
log_interval = 1
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# Wandb logging
wandb_log = False # disabled by default
wandb_project = 'shakespeare'
wandb_run_name = 'kronzo-shakespeare'

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context length

# Model (small GPT)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1 # for finetuning
bias = False # do we use bias inside LayerNorm and Linear layers?

# Optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 8000 # total number of training iterations
weight_decay = 1e-1
# For KronZO, beta1 and beta2 are primarily for AdamW if used as a fallback,
# or for KronZO's own momentum if enabled via use_momentum.
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# Learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 500 # how many steps to warm up for
lr_decay_iters = 8000 # should be ~= max_iters
min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10

# Training method
train_method = 'kronzo'

# KronZO specific parameters
zo_eps = 1e-3                    # Perturbation size
kron_strategy = 'approx_square'  # Kronecker factorization strategy ('approx_square', 'fixed_factor', 'power2')
kron_max_factor = 32             # Maximum factor size for 'fixed_factor' strategy

# KronZO Momentum settings (optional)
use_momentum = False             # Set to True to enable momentum for KronZO
momentum_beta = 0.9              # Momentum coefficient (if use_momentum is True)

# Adaptive zo_eps settings (optional)
use_adaptive_eps = False         # Enable adaptive zo_eps for KronZO
adaptive_eps_window = 20
adaptive_eps_lr_coupling = 0.3
adaptive_eps_success_high = 0.8
adaptive_eps_success_low = 0.2

# System settings
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster 
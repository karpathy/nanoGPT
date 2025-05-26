# Config for KronZO training on OpenWebText dataset

import torch

# Output directory
out_dir = 'out-kronzo-owt'

# Logging settings
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# Wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'kronzo-owt'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 40 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# Model (GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# Optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
# For KronZO, beta1 and beta2 are primarily for AdamW if used as a fallback,
# or for KronZO's own momentum if enabled via use_momentum.
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# Learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10

# Training method
train_method = 'kronzo'

# KronZO specific parameters
zo_eps = 1e-3                    # Perturbation size
kron_strategy = 'approx_square'  # Kronecker factorization strategy ('approx_square', 'fixed_factor', 'power2')
kron_max_factor = 64             # Maximum factor size for 'fixed_factor' strategy (larger for bigger model)

# KronZO Momentum settings (optional)
use_momentum = True              # Enable momentum for better convergence on large dataset
momentum_beta = 0.9              # Momentum coefficient

# Adaptive zo_eps settings (optional)
use_adaptive_eps = True          # Enable adaptive zo_eps for better performance
adaptive_eps_window = 50
adaptive_eps_lr_coupling = 0.5
adaptive_eps_success_high = 0.8
adaptive_eps_success_low = 0.2

# System settings
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster 
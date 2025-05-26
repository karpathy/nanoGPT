# Config for DiLoZO with rank adaptive training on GPT-2 pretraining dataset

# Output directory
out_dir = 'out-dilozo-rank-adaptive-gpt2'

# Logging settings
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Wandb logging
wandb_log = False
wandb_project = 'owt'
wandy_run_name = 'dilozo-rank-adaptive-gpt2'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 40
batch_size = 12
block_size = 1024

# Model (GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# Training method
train_method = 'dilozo'

# DiLoZO specific parameters
zo_eps = 1e-3
directional_q = 10
# rank_r is the initial/fixed rank if rank_adaptive is False
rank_r = 16 
step_interval = 20

# DiLoZO Momentum settings
use_momentum = False
momentum_beta = 0.9

# Rank-adaptive settings
rank_adaptive = True     # Enable adaptive rank scheduling
min_rank = 4             # Minimum rank for adaptive scheduling
max_rank = 32            # Maximum rank for adaptive scheduling

# Adaptive zo_eps settings
use_adaptive_eps = False
adaptive_eps_window = 50
adaptive_eps_lr_coupling = 0.5
adaptive_eps_success_high = 0.7
adaptive_eps_success_low = 0.3
dimezo_direct_movement = False # Not used by DiLoZO

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
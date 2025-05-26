# Config for DiLoZO with Adaptive zo_eps training on Shakespeare dataset

# Output directory
out_dir = 'out-dilozo-adaptive-shakespeare'

# Logging settings
eval_interval = 500
log_interval = 1
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Wandb logging
wandb_log = False
wandb_project = 'shakespeare'
wandy_run_name = 'dilozo-adaptive-shakespeare'

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Model (small GPT)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# Optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 500
lr_decay_iters = 5000
min_lr = 1e-4

# Training method
train_method = 'dilozo'

# DiLoZO specific parameters
zo_eps = 1e-3             # Base perturbation size (will be adapted)
directional_q = 10
rank_r = 4
step_interval = 10

# DiLoZO Momentum settings
use_momentum = False
momentum_beta = 0.9

# Rank-adaptive settings
rank_adaptive = False
min_rank = 1
max_rank = 16

# Adaptive zo_eps settings
use_adaptive_eps = True  # Enable adaptive zo_eps for DiLoZO
adaptive_eps_window = 20
adaptive_eps_lr_coupling = 0.3
adaptive_eps_success_high = 0.8
adaptive_eps_success_low = 0.2
dimezo_direct_movement = False # Not used by DiLoZO

# System settings
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True 
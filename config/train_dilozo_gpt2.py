# Config for DiLoZO training on GPT-2 pretraining dataset (e.g., OpenWebText)

# Output directory
out_dir = 'out-dilozo-gpt2'

# Logging settings
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' (e.g., 'gpt2-medium')

# Wandb logging
wandb_log = False # Set to True for wandb logging
wandb_project = 'owt' # Or your project name
wandy_run_name = 'dilozo-gpt2'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 40  # Corresponds to a total batch size of 40*12 = 480
batch_size = 12                 # Micro-batch size
block_size = 1024               # Context length

# Model (GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # For pretraining, 0 is good
bias = False

# Optimizer
learning_rate = 6e-4  # Max learning rate
max_iters = 600000    # Total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000 # Should be ~= max_iters
min_lr = 6e-5         # Minimum learning rate

# Training method
train_method = 'dilozo'

# DiLoZO specific parameters
zo_eps = 1e-3
directional_q = 10 # Number of directions for DiLoZO
rank_r = 16        # Rank for U and V matrices (can be tuned)
step_interval = 20 # Interval for updating V matrices (can be tuned)

# DiLoZO Momentum settings
use_momentum = False
momentum_beta = 0.9

# Rank-adaptive settings
rank_adaptive = False
min_rank = 4
max_rank = 32 # Example values, tune as needed

# Adaptive zo_eps settings
use_adaptive_eps = False
adaptive_eps_window = 50 # Larger window for larger scale experiments
adaptive_eps_lr_coupling = 0.5
adaptive_eps_success_high = 0.7
adaptive_eps_success_low = 0.3
dimezo_direct_movement = False # Not used by DiLoZO

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
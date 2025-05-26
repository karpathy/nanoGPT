# Config for Rank-Adaptive LOZO training on Shakespeare dataset
# This demonstrates the rank scheduling from min_rank to max_rank

# Output directory
out_dir = 'out-rank-adaptive-lozo-shakespeare'

# Logging settings
eval_interval = 1000
log_interval = 1
eval_iters = 50
eval_only = False

# Training data
dataset = 'shakespeare'
gradient_accumulation_steps = 1  # must be divisible by number of GPUs
batch_size = 64
block_size = 256  # context length

# Model configuration (small GPT)
n_layer = 6
n_head = 6
n_embd = 384  # embedding dimension
dropout = 0.1
bias = False

# Learning rate settings
learning_rate = 1e-3  # standard learning rate for LOZO
max_iters = 5000
weight_decay = 1e-1
# grad_clip = 1.0  # clip gradients at this value

# Set beta1 and beta2 (not used by LOZO, but needed for compatibility)
beta1 = 0.9
beta2 = 0.99

# Learning rate schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 5000
min_lr = 1e-4

# Training method
train_method = 'lozo'  # Use LOZO optimizer

# LOZO specific parameters
zo_eps = 1e-3       # perturbation size for zero-order gradient estimation
rank_r = 4          # base rank (used when rank_adaptive=False)
step_interval = 10  # interval for updating the V matrices (every ν steps)
use_momentum = False # whether to use momentum in LOZO (LOZO-M)
momentum_beta = 0.9 # momentum coefficient for LOZO-M
zo_q = 10            # number of finite differences computations to average over

# Rank-adaptive LOZO parameters
rank_adaptive = True # Enable adaptive rank scheduling
min_rank = 1        # Starting rank
max_rank = 16       # Ending rank (will reach this at max_iters)

# System settings
device = 'cuda'  # use 'cpu' or 'mps' if needed
dtype = 'bfloat16'  # bfloat16 might not be available on all devices
compile = True  # set to False for CPU training 
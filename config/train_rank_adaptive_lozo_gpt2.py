# Config for Rank-Adaptive LOZO pretraining of GPT-2 (124M) model
# Demonstrates rank scheduling from min_rank=1 to max_rank=16 across training

# Output directory
out_dir = 'out-rank-adaptive-lozo-gpt2'

# Logging settings
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False

# Training data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # simulate larger batch sizes
batch_size = 12  # micro-batch size
block_size = 1024  # context length

# Model configuration (standard GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False  # don't use bias in LayerNorm and Linear layers

# Learning rate settings
learning_rate = 6e-4  # max learning rate for LOZO
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
grad_clip = 1.0  # clip gradients at this value

# Set beta1 and beta2 (not used by LOZO, but needed for compatibility)
beta1 = 0.9
beta2 = 0.95

# Learning rate schedule
decay_lr = True
warmup_iters = 2000  # warmup steps
lr_decay_iters = 600000
min_lr = 6e-5  # minimum learning rate

# Training method
train_method = 'lozo'  # Use LOZO optimizer

# LOZO specific parameters
zo_eps = 1e-3       # perturbation size for zero-order gradient estimation
rank_r = 4          # base rank (used when rank_adaptive=False)
step_interval = 10  # interval for updating the V matrices (every ν steps)
use_momentum = False # whether to use momentum in LOZO (LOZO-M)
momentum_beta = 0.9 # momentum coefficient for LOZO-M
zo_q = 1            # number of finite differences computations to average over

# Rank-adaptive LOZO parameters
rank_adaptive = True # Enable adaptive rank scheduling
min_rank = 1        # Starting rank (very low-rank at beginning)
max_rank = 16       # Ending rank (higher rank towards end of training)

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
# Config for SVD-LOZO with Full SVD training on Shakespeare dataset
# This tests the full SVD implementation

# Output directory
out_dir = 'out-svdlozo-full-shakespeare'

# Logging settings
eval_interval = 500
log_interval = 1
eval_iters = 50
eval_only = False

# Training data
dataset = 'shakespeare'
gradient_accumulation_steps = 1  # no need for gradient accumulation on smaller dataset
batch_size = 64
block_size = 256  # context length

# Model configuration (small GPT)
n_layer = 6
n_head = 6
n_embd = 384  # embedding dimension
dropout = 0.1
bias = False

# Learning rate settings
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Learning rate schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 5000
min_lr = 1e-4

# Training method
train_method = 'svdlozo'  # Use SVD-LOZO optimizer

# SVD-LOZO specific parameters 
zo_eps = 1e-3  # perturbation size
svd_tau = 0.6  # threshold for adaptive rank selection (keep σ > τ * σ_max) - increased for actual rank reduction
svd_max_rank = 16  # maximum rank for SVD
use_full_svd = True  # Use FULL SVD instead of randomized SVD 
zo_q = 1

# System settings
device = 'cuda'  # use 'cpu' or 'mps' if needed
dtype = 'bfloat16'  # bfloat16 might not be available on all devices
compile = True  # set to False for CPU training 
# Config for LOZO pretraining of GPT-2 (124M) model

# Output directory
out_dir = 'out-lozo-gpt2'

# Logging settings
eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False

# Training data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # simulate larger batch sizes
batch_size = 32  # micro-batch size
block_size = 1024  # context length

# Model configuration (standard GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False  # don't use bias in LayerNorm and Linear layers

# Learning rate settings - optimized for LOZO
learning_rate = 1e-6
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Learning rate schedule
decay_lr = False
warmup_iters = 4000
lr_decay_iters = 100000
min_lr = 1e-6

# LOZO specific parameters
zo_eps = 1e-3  # perturbation size
rank_r = 2  # rank for low-rank perturbation (increased from original value)
step_interval = 50  # interval for updating the V matrices
use_momentum = False  # disable momentum for standard LOZO

# Training method
train_method = 'lozo'  # Use LOZO optimizer

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
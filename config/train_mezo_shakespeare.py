# Config for MeZO training on Shakespeare dataset
# This is a smaller model configuration for quick testing

# Output directory
out_dir = 'out-mezo-shakespeare'

# Logging settings
eval_interval = 2500
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

# MeZO specific parameters 
zo_eps = 1e-3  # perturbation size for zero-order gradient estimation
zo_q = 11  # number of finite difference estimates to average (default: 1)
use_momentum = False  # disable momentum for standard MeZO

# Training method
train_method = 'mezo'  # Use MeZO optimizer

# System settings
device = 'cuda'  # use 'cpu' or 'mps' if needed
dtype = 'bfloat16'  # bfloat16 might not be available on all devices
compile = True  # set to False for CPU training 
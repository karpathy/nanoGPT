# Config for MeZO-M training on Shakespeare dataset
# This is a smaller model configuration for quick testing with momentum

# Output directory
out_dir = 'out-mezom-shakespeare'

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
learning_rate = 1e-3  # slightly lower learning rate due to momentum
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
zo_q = 50
use_momentum = True  # enable momentum for MeZO-M
momentum_beta = 0.9  # momentum coefficient

# Training method
train_method = 'mezom'  # Use MeZO-M optimizer

# System settings
device = 'cuda'  # use 'cpu' or 'mps' if needed
dtype = 'bfloat16'  # bfloat16 might not be available on all devices
compile = True  # set to False for CPU training 
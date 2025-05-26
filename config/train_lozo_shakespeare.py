# Config for LOZO training on Shakespeare dataset
# This is a smaller model configuration for quick testing

# Output directory
out_dir = 'out-lozo-shakespeare'

# Logging settings
eval_interval = 1000
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
learning_rate = 1e-10
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Learning rate schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 5000
min_lr = 1e-11

train_method = 'lozo'  # Use MeZO optimizer

# LOZO specific parameters 
zo_eps = 1e-3  # perturbation size
rank_r = 4  # rank for low-rank perturbation
step_interval = 50  # interval for updating the V matrices (every ν steps)
zo_q = 1

# System settings
device = 'cuda'  # use 'cpu' or 'mps' if needed
dtype = 'bfloat16'  # bfloat16 might not be available on all devices
compile = True  # set to False for CPU training 
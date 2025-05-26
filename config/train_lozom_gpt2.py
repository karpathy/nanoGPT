# Config for LOZO-M pretraining of GPT-2 (124M) model

# Output directory
out_dir = 'out-lozom-gpt2'

# Logging settings
eval_interval = 100
log_interval = 1
eval_iters = 100
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

# Learning rate settings - lower for LOZO-M
learning_rate = 1e-3  # lower learning rate is better with momentum
max_iters = 20000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Learning rate schedule
decay_lr = True
warmup_iters = 500  # increased from 2000 for better stability
lr_decay_iters = 20000
min_lr = 1e-4  # minimum learning rate

# LOZO specific parameters
zo_eps = 1e-3  # perturbation size
rank_r = 4  # rank for low-rank perturbation
step_interval = 50  # interval for updating the V matrices
zo_q = 10
# LOZO-M specific parameters
use_momentum = True  # enable momentum
momentum_beta = 0.9  # momentum coefficient

# Training method
train_method = 'lozom'  # Use LOZO-M optimizer

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
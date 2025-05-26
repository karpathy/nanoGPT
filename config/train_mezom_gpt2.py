# Config for MeZO-M (with momentum) pretraining of GPT-2 (124M) model

# Output directory
out_dir = 'out-mezom-gpt2'

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

# Learning rate settings - optimized for MeZO with momentum
learning_rate = 1e-5  # learning rate for MeZO-M
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Learning rate schedule
decay_lr = False
warmup_iters = 4000  # warmup steps
lr_decay_iters = 100000
min_lr = 1e-6  # minimum learning rate

# MeZO specific parameters
zo_eps = 5e-4  # perturbation size for zero-order gradient estimation
use_momentum = True  # enable momentum for MeZO-M
momentum_beta = 0.9  # momentum coefficient

# Training method
train_method = 'mezom'  # Use the mezo method with momentum

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
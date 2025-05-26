# Config for DiMeZO with Adaptive zo_eps training on Shakespeare dataset
# This tests our new directional selection with adaptive perturbation size

# Output directory
out_dir = 'out-dimezo-adaptive-shakespeare'

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
train_method = 'dimezo'  # Use DiMeZO optimizer

# DiMeZO specific parameters 
zo_eps = 1e-3  # base perturbation size (will be adapted)
directional_q = 10  # number of directions to try in directional selection (DiMeZO/DiLoZO)
dimezo_direct_movement = False  # Use traditional gradient estimation: (f+ - f-)/2ε
zo_q = 1       # not used for DiMeZO but kept for compatibility

# Adaptive zo_eps parameters
use_adaptive_eps = True     # Enable adaptive zo_eps
adaptive_eps_window = 20    # Smaller window for faster adaptation on small dataset
adaptive_eps_lr_coupling = 0.3  # More conservative coupling between eps and learning rate
adaptive_eps_success_high = 0.8  # Higher threshold for increasing eps (be more conservative)
adaptive_eps_success_low = 0.2   # Lower threshold for decreasing eps

# System settings
device = 'cuda'  # use 'cpu' or 'mps' if needed
dtype = 'bfloat16'  # bfloat16 might not be available on all devices
compile = True  # set to False for CPU training 
# Config for DiMeZO with Adaptive zo_eps training on GPT-2 pretraining dataset
# This tests our new directional selection with adaptive perturbation size on larger scale

# Output directory
out_dir = 'out-dimezo-adaptive-gpt2'

# Logging settings
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True

# Training data
dataset = 'openwebtext'
gradient_accumulation_steps = 40  # simulate larger batch sizes
batch_size = 12  # micro-batch size
block_size = 1024  # context length

# Model configuration (GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good
bias = False

# Learning rate settings
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# Training method
train_method = 'dimezo'  # Use DiMeZO optimizer

# DiMeZO specific parameters 
zo_eps = 1e-3  # base perturbation size (will be adapted)
dimezo_q = 10  # number of directions to try in directional selection
dimezo_direct_movement = False  # Use traditional gradient estimation: (f+ - f-)/2ε
zo_q = 1       # not used for DiMeZO but kept for compatibility

# Adaptive zo_eps parameters
use_adaptive_eps = True     # Enable adaptive zo_eps
adaptive_eps_window = 30    # Standard window for large-scale training
adaptive_eps_lr_coupling = 0.3  # Conservative coupling between eps and learning rate
adaptive_eps_success_high = 0.8  # Higher threshold for increasing eps (be conservative)
adaptive_eps_success_low = 0.2   # Lower threshold for decreasing eps

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
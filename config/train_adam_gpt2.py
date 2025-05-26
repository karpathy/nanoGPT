# Config for Adam optimizer pretraining of GPT-2 (124M) model

# Output directory
out_dir = 'out-adam-gpt2'

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

# Adam optimizer settings
learning_rate = 6e-4  # max learning rate for Adam
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value

# Learning rate schedule
decay_lr = True
warmup_iters = 2000  # warmup steps
lr_decay_iters = 100000
min_lr = 6e-5  # minimum learning rate

# Training method
train_method = 'adam'  # Use Adam optimizer

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 
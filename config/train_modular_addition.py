# Configuration for a modular arithmetic training

out_dir = "out-modular-arithmetic"
eval_interval = 5  # frequency increased for train vs val monitoring
eval_iters = 200
log_interval = 5 # decreased for higher resolution

always_save_checkpoint = False # this just says it saves whenever val improves
only_save_checkpoint_at_end = True # this only saves at the end of training, speeding things up

# logging
log_project = "out-modular-addition"
log_run_name = "logs-modular-addition"

# tensorboard
tensorboard_log = True
tensorboard_project = log_project
tensorboard_run_name = log_run_name

# wandb
wandb_log = False
wandb_project = log_project
wandb_run_name = log_run_name

dataset = "modular_addition"
gradient_accumulation_steps = 1
batch_size = 64

# Change to be `modulo + 1` if no-separator, or `modulo + 3` if there are separators
block_size = 7

# Model parameters
n_layer = 4
n_head = 4
n_embd = 64
dropout = 0.4

# Training parameters
learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Uncomment the lines below if running on a MacBook without GPU
# device = 'cpu'
# compile = False

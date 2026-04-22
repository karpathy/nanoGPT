eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = 'nanoGPT-dissertation'

init_from = 'gpt2'
dataset = 'gsm8k'

# T4 settings
batch_size = 6
block_size = 1024
gradient_accumulation_steps = 10
device = 'cuda'
dtype = 'float16'
compile = False

# optimizer
learning_rate = 3e-5
max_iters = 2000
lr_decay_iters = 5000
min_lr = 3e-6
beta2 = 0.99
warmup_iters = 200
weight_decay = 1e-1
grad_clip = 1.0

# prefix defaults — all overridden by CLI per run
prefix_len = 0
prefix_update_period = 1
prefix_cache = False
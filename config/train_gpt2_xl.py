import os

wandb_log = True
wandb_project = 'owt'
wandb_run_name=f'gpt2-1.5B-4-node'
out_dir = f'out/{wandb_run_name}'
os.makedirs(out_dir, exist_ok=True)

n_layer = 48
n_head = 25
n_embd = 1600

batch_size = 27
block_size = 1024
gradient_accumulation_steps = 32

# 5 TPP
max_iters = 10466 
warmup_iters = 0.03 * max_iters
lr_decay_iters = max_iters

# eval stuff
eval_interval = 1000
eval_iters = 10
log_interval = 1

# weight decay
weight_decay = 1e-1

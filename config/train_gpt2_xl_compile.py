# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 60 batch size * 1024 block size * 8 GPUs = 491,520
batch_size = 48
block_size = 1024
gradient_accumulation_steps = 8

# n_layer = 12
# n_head = 12
# n_embd = 768

# gpt2 -xl
n_layer = 48
n_head = 25
n_embd = 1600

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 100000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

compile = True
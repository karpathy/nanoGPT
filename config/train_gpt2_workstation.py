# Specialization to my personal workstation paths

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = experiment_name = wandb_run_name = 'gpt2-124M-normalized_gpt'
wandb_log = True
wandb_notes = "Base normalized GPT run"
wandb_project = "normalized_gpt_dev_sakle"

data_root_path='/mnt/data/'
dataset = 'nanoGPTopenweb'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
learning_rate = 1e-3
warmup_iters = 0 # how many steps to warm up for

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 0.0

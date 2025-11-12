# config for training GPT-2 (124M) with ALiBi down to very nice loss
# This config trains on sequences of length 1024 but can extrapolate to longer sequences at inference
# launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_alibi.py

wandb_log = True
wandb_project = 'owt-alibi'
wandb_run_name='gpt2-124M-alibi'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# ALiBi configuration
use_alibi = True  # Enable ALiBi attention for length extrapolation

# Note: With ALiBi, the model can handle longer sequences at inference time
# without additional training or fine-tuning
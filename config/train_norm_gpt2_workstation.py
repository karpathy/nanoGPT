# Specialization to my personal workstation paths

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
from model import nGPT, nGPTConfig
model_class = nGPT
model_config = nGPTConfig

out_dir = experiment_name = wandb_run_name = 'gpt2-124M-normalized_gpt'
wandb_log = True
wandb_notes = "Base normalized GPT run"
wandb_project = "normalized_gpt_dev_sakle"

data_root_path='/mnt/data/'
dataset = 'nanoGPTopenweb'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 1 # 5 * 8
base_scale_override = None # set to None to default to normalized GPT initialization

compile = False
# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
learning_rate = 15e-4
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
min_lr = 0.0
weight_decay = 0.0
warmup_iters = 0

# weight decay
weight_decay = 0.0

# Model dimension settings
n_layer = 12
n_head = 16
n_embd = 1024
base_scale_override = 1.0 / n_embd ** 0.5
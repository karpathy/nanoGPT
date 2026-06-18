#  T4-optimised finetuning config

# Finetune GPT-2 124M on OWT — tuned for single T4 16GB
# Replaces train_gpt2.py which requires 8x A100

out_dir = '/kaggle/working/gpt2_owt_baseline'
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

# WandB — same project/run name as before
wandb_log = True
wandb_project = 'nanoGPT-dissertation'
wandb_run_name = 'gpt2-owt-kaggle-baseline'

# Init from pretrained GPT-2 (finetuning, not scratch)
init_from = 'gpt2'

# Data
dataset = 'openwebtext'

# ---- MEMORY: this is the only thing that killed your run ----
batch_size = 6              # was 12 → halved
block_size = 512            # was 1024 → halved (4x memory saving)
gradient_accumulation_steps = 10   # keeps effective batch ~= 30k tokens
# effective = 6 * 512 * 10 = 30,720 tokens/step  (vs 491,520 before)

# Model — must match GPT-2 exactly since init_from='gpt2'
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Optimizer — lower LR since we are finetuning, not pretraining
learning_rate = 3e-5        # was 6e-4 in train_gpt2.py
max_iters = 5000
lr_decay_iters = 5000
min_lr = 3e-6
beta2 = 0.99
warmup_iters = 200
weight_decay = 1e-1
grad_clip = 1.0

# System
device = 'cuda'
dtype = 'float16'           # T4 has no bfloat16 support — keeps compilation clean
compile = False             # avoids the bfloat16 compile warning + is safer on T4


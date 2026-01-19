# GPT-2 Small training on Shakespeare with BPE tokenizer
# Configuration aligned with tt-train/configs/training_configs/training_shakespeare_gpt2.yaml
# and tt-train/configs/model_configs/gpt2s.yaml

out_dir = 'out-shakespeare-gpt2s'
eval_interval = 5000  # disable evaluation during training for fair timing comparison
eval_iters = 200
log_interval = 1  # log every iteration for comparison

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-gpt2s'
wandb_run_name = 'gpt2s'

# Data - uses GPT-2 BPE tokenizer (vocab_size=50257)
dataset = 'shakespeare'

# use LoRA for attention
lora_attention = False

# Training config (matching tt-train)
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 5000
block_size = 1024  # max_sequence_length

# Model config (GPT-2 Small, matching tt-train gpt2s.yaml)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2  # tt-train uses dropout_prob: 0.2
bias = True  # tt-train uses bias by default in LinearLayer

# Optimizer config (matching tt-train AdamW defaults)
learning_rate = 3e-4  # tt-train: 0.0003
weight_decay = 0.1  # tt-train: 0.1
beta1 = 0.9  # tt-train default
beta2 = 0.999  # tt-train default (NOT 0.95 like nanoGPT default)

# Disable gradient clipping (tt-train: use_clip_grad_norm: false)
grad_clip = 0.0

# Disable LR decay (tt-train uses identity scheduler)
decay_lr = False

# These are ignored when decay_lr = False, but set for completeness
warmup_iters = 0
lr_decay_iters = 5000
min_lr = 3e-5

# Compile for speed (can disable for debugging)
compile = True


# Train a baseline model (standard attention) for comparison with Attention Sink
# This config uses standard causal attention (no attention sink)
# Model architecture is identical to attention sink variant for fair comparison

out_dir = 'out-baseline-sink'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Save checkpoints for comparison
always_save_checkpoint = True

wandb_log = True
wandb_project = 'attention-sink-experiment'
wandb_run_name = 'minigpt-baseline'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context window

# Baby GPT model (same as attention sink variant)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Standard attention (no attention sink)
use_attention_sink = False

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# For CPU testing:
# device = 'cpu'
# compile = False

# Train a model with Attention Sink mechanism
# Experiment based on "Efficient Streaming Language Models with Attention Sinks"
# This config enables attention sink with sliding window attention

out_dir = 'out-attention-sink'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Save checkpoints for comparison
always_save_checkpoint = True

wandb_log = True
wandb_project = 'attention-sink-experiment'
wandb_run_name = 'minigpt-attention-sink'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context window

# Baby GPT model (same as baseline for fair comparison)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Attention Sink parameters
use_attention_sink = True
sink_size = 4  # Keep first 4 tokens as attention sinks
window_size = 252  # Sliding window of 252 tokens (total = 4 + 252 = 256)

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# For CPU testing:
# device = 'cpu'
# compile = False

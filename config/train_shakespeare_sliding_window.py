# config for training a GPT with sliding window attention on Shakespeare data
# demonstrates the new sliding window attention feature

# training parameters
out_dir = 'out-shakespeare-sliding-window'
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-sliding-window'
wandb_run_name = 'mini-gpt-sliding-window'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# baby GPT model with sliding window attention
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# SLIDING WINDOW ATTENTION
# Set window_size to limit attention to recent tokens only
# None = full attention (default), int = sliding window with specified size
# For example, window_size=128 means each token can only attend to the
# previous 128 tokens, making the model more efficient for long sequences
window_size = 128  # Each token attends to at most 128 previous tokens

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4

beta1 = 0.9
beta2 = 0.95

warmup_iters = 100

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

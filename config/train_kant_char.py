# config/train_kant_char.py
out_dir = 'out-kant-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'kant-char'
wandb_run_name = 'mini-gpt'

dataset = 'kant'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4

beta2 = 0.99
warmup_iters = 100
# config/train_calls.py

out_dir = 'out-calls'
eval_interval = 100
eval_iters = 20
log_interval = 10

always_save_checkpoint = True
wandb_log = False

dataset = 'calls'  # must match your `data/calls` folder
gradient_accumulation_steps = 1
batch_size = 4
block_size = 128  # you can try 256 or 512 later

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

bias = False  # use bias in linear layers
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

compile = False  # if True, uses torch.compile (PyTorch 2.0+)

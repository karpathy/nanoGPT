out_dir = 'out-bench-baseline'
eval_interval = 500
eval_iters = 20
log_interval = 100

always_save_checkpoint = False
wandb_log = False
dataset = 'shakespeare_char'

gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cpu'
compile = False

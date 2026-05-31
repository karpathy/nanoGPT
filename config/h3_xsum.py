eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = 'nanoGPT-dissertation'

init_from = 'gpt2'
dataset = 'xsum'

batch_size = 4
block_size = 1024
gradient_accumulation_steps = 8
device = 'cuda'
dtype = 'float16'
compile = False

learning_rate = 0.03
max_iters = 2000
lr_decay_iters = 2000
min_lr = 3e-6
beta2 = 0.99
warmup_iters = 200
weight_decay = 1e-1
grad_clip = 1.0

prefix_type = 'soft'
prefix_len = 0
prefix_update_period = 1
prefix_cache = False

generation_eval = 'none'
run_final_em = False
out_dir = 'out-transcendence-gpt'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'transcendence-gpt'
wandb_run_name = 'in-distribution'

dataset = 'card_set'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 540 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

vocab_size = 17

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

dtype = "float32"
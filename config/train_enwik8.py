# train a character-level model on enwik8

out_dir = "out-enwik8"
eval_interval = 1000
eval_iters = 200
log_interval = 100  # don't print too too often

# only save when val improves
always_save_checkpoint = False

# wandb_log = True # override via command line if you like
# wandb_project = 'nanogpt'
# wandb_run_name = 'enwik8'

dataset = "enwik8"
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 512
dropout = 0.2

learning_rate = 5e-4
max_iters = 100000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 5e-5  # learning_rate / 10 usually
beta2 = 0.99

warmup_iters = 200  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = True  # do not torch compile the model
# init_from = 'resume'
# eval_only = True

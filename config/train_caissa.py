# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'output'
eval_interval = 2000
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'lichess-puzzle-positions-large-sample'
wandb_run_name = 'large-caissa'

dataset = 'chess-data'
gradient_accumulation_steps = 4
batch_size = 128 # a single puzzle is max 81 tokens, adding for extra padding
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
# made larger
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 1e6
lr_decay_iters = 1e6 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
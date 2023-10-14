# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-sat'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'sat-solve-gpt'
wandb_run_name = 'sat-gpt'

dataset = 'SAT'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 1289 # Note: longest sequence is 1290, -1 for prediction

# baby GPT model for first experiments
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

max_iters = 10000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

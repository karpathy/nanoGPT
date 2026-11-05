# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-eeg-single'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'nanogpt-eeg'
wandb_run_name = 'eeg-single-vanilla'

dataset = 'eeg_single'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

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

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

pos_encoding = 'sinusoidal' #jm
use_codebook = True   #jm tokens enter as their DAC codebook vectors (data/<dataset>/codebook.npy)
eeg_dataloader = True   #jm windows from the per-recording grids
bidirectional = True   #jm
mask_ratio = 0.15      #jm
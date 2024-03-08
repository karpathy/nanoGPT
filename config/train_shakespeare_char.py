# train a miniature character-level shakespeare model

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# optimizer parameters
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = True 
lr_schedule = 'None'
opt_type = 'adam'
weight_decay = 1e-1
learning_rate = 1e-5 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = learning_rate/10 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially
compile = False # do not torch compile the model
#device = 'cpu'  # run on cpu only

# logging
wandb_log = True # override via command line if you like
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often
always_save_checkpoint = False # we expect to overfit on this small dataset, so only save when val improves

wandb_project = 'shakespeare-adam'
dataset = 'shakespeare_char'
out_dir = 'out-shakespeare-char'

wandb_run_name = \
        '{}  | '.format(opt_type) + \
        'lr: {:.2e}-{:.2e} | '.format(min_lr, learning_rate) + \
        'weight-decay: {:.2e}  | '.format(weight_decay) + \
        'lr-decay: {}-{}  | '.format(decay_lr, lr_decay_iters) + \
        'beta1: {} beta2: {}  | '.format(beta1, beta2) + \
        'warmup: {}  | '.format(warmup_iters) + \
        'lr-scheduler: {}  | '.format(lr_schedule) + \
        'max-iters: {}'.format(max_iters)


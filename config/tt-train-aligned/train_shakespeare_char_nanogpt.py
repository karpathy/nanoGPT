# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 5000 # keep frequent because we'll overfit
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

# use LoRA for attention
lora_attention = False

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# == Below are the original settings from nanoGPT repository == 
# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# == This is currently used cofiguration in tt-train == 
learning_rate = 3e-4
weight_decay = 0.01
decay_lr = False
grad_clip = 0.0
beta2 = 0.999

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

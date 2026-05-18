# train a miniature character-level shakespeare+wiki model
# combined corpus is ~2-3x tinyshakespeare, so we train for ~50% more iters.
# block_size / n_layer / n_head / n_embd kept identical to train_shakespeare_char.py
# so the run still fits in a Colab T4 in roughly the same wall-clock ballpark.

out_dir = 'out-shakespeare-wiki-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-wiki-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_wiki_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 12000
lr_decay_iters = 12000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

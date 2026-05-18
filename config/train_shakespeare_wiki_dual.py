# train a miniature character-level shakespeare+wiki model with DUAL WEIGHT SLOTS
#
# Same model size as train_shakespeare_wiki_char.py per slot; ~2x params total
# from holding W_s and W_w. Same alternating-pass cadence as train_diff_logging.py
# (iters_per_pass=73 over 12000 iters = 164 passes).
#
# Dual-trainer knobs:
#   mix_distribution = 'beta_half'     -> alpha ~ Beta(0.5, 0.5), beta = 1 - alpha
#   sample_alpha_beta_every = 1        -> resample at every iter

out_dir = 'out-shakespeare-wiki-dual'
eval_interval = 250
eval_iters = 200
log_interval = 10

dataset = 'shakespeare_wiki_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 12000
lr_decay_iters = 12000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# per-pass alternation
iters_per_pass = 73
first_pass_corpus = 'shake'

# dual-slot mixing — alpha + beta = 1, alpha ~ Beta(0.5, 0.5)
mix_distribution = 'beta_half'
sample_alpha_beta_every = 1

# on macbook also add
# device = 'cpu'
# compile = False

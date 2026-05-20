# Karpathy-matched dual config on shake+TinyStories char.
#
# Per-slot architecture identical to nanoGPT's config/train_shakespeare_char.py
# (n_layer=6, n_head=6, n_embd=384, dropout=0.2, lr=1e-3, beta2=0.99) so each
# slot at its native corner is doing the same task Karpathy's 10.65M model
# solved. Whatever gap shows up between the shake-corner val and Karpathy's
# 1.4697 reference is the measurable cost of dual training.
#
# Reference framing: the model the user samples from at every alpha is
# ~10.65M params (W = alpha*W_s + (1-alpha)*W_w is a single weight tensor
# of the same shape Karpathy's model has). Total stored params are ~2x
# because both slot endpoints have to be held to define the slider; that
# is the storage cost of a continuous source-attribution slider, not a
# capacity bump.
#
# Iter budget: Karpathy's 5000 iters are all shake-batch optimizer steps.
# Under alt_mixed Uniform, W_s only steps every other pass, so 10000 total
# iters gives W_s the same 5000 effective updates Karpathy used. lr_decay
# proportionally extended.
#
# Trained with alt_mixed Uniform — the winning recipe from the wiki
# experiments (clean per-slot Adam isolation + mixed-batch forwards).

out_dir = 'out-shakespeare-tinystories-dual-karpathy'
eval_interval = 250
eval_iters = 200
log_interval = 20

dataset = 'shakespeare_tinystories_char'
gradient_accumulation_steps = 1   # Karpathy ran no grad accum; effective batch 64
batch_size = 64
block_size = 256

# KARPATHY-MATCHED GPT (per slot)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

iters_per_pass = 73
first_pass_corpus = 'shake'

mix_distribution = 'uniform'
sample_alpha_beta_every = 1

batch_mode = 'alt_mixed'

gradient_normalize = False
grad_norm_floor = 0.05

save_every_passes = 7

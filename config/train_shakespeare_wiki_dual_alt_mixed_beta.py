# alt_mixed mode with Beta_half alpha sampling.
#
# Same as train_shakespeare_wiki_dual_alt_mixed.py but mix_distribution =
# 'beta_half' instead of 'uniform'. The hypothesis: alternating mode's
# coherent-at-corner behavior came from Beta_half concentrating training
# mass at alpha=0,1 (training the slots as standalone models, not as
# blend-with-each-other models). Uniform alpha samples spread training
# across the [0, 1] range, so the slot's training is mostly at alpha=0.5,
# pushing it toward "good in blend" rather than "good standalone."
#
# This config tests whether alt_mixed + Beta_half recovers the standalone-
# corner behavior while keeping the two-optimizer isolation.

out_dir = 'out-shakespeare-wiki-dual-alt-mixed-beta'
eval_interval = 250
eval_iters = 200
log_interval = 10

dataset = 'shakespeare_wiki_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

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

iters_per_pass = 73
first_pass_corpus = 'shake'

# THE DIFF vs train_shakespeare_wiki_dual_alt_mixed.py: Beta_half places
# training mass at alpha=0,1 corners so each slot's gradient signal during
# its training pass is mostly "improve as a standalone model."
mix_distribution = 'beta_half'
sample_alpha_beta_every = 1

batch_mode = 'alt_mixed'

gradient_normalize = False
grad_norm_floor = 0.05

save_every_passes = 7

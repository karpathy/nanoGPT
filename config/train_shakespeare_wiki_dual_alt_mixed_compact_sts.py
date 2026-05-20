# Compact char-level dual model for shake+TinyStories (sts).
#
# Same training scheme as train_shakespeare_wiki_dual_alt_mixed.py
# (alt_mixed batch_mode + Uniform alpha sampling + two AdamW optimizers
# alternating at the pass level), but on the new
# `shakespeare_tinystories_char` dataset and with a deliberately
# SMALLER model.
#
# Why smaller:
#   * TinyStories is far simpler than the wiki dump used by
#     shakespeare_wiki_char — tight vocab, short declarative sentences,
#     heavily repeated structure. Karpathy's stories260K shows even a
#     260K-param model produces coherent TinyStories output.
#   * The 21M dual (n_layer=6, n_head=6, n_embd=384) on shake+wiki hit
#     val ~1.78 at 12000 iters. On shake+tinystories that ceiling is
#     way lower (expected ~1.3–1.4) and a compact model gets there fast.
#
# Sizing: n_layer=4, n_head=4, n_embd=256, block_size=256.
# Each slot is ~3.2M params; dual model totals ~6.4M.
#
# Schedule: max_iters=6000, lr_decay_iters=6000. At 73 iters/pass,
# 6000 / (2 * 73) ≈ 41 full alternating pass-pairs — plenty for
# convergence on TinyStories at char level.
#
# Dropout: 0.05 (down from the wiki config's 0.2). The wiki runs taught
# us 0.2 is over-regularizing in alt_mixed mode — each slot already gets
# implicit regularization from being co-mixed with the other in every
# forward pass.

out_dir = 'out-shakespeare-tinystories-dual-alt-mixed-compact'
eval_interval = 250
eval_iters = 200
log_interval = 10

dataset = 'shakespeare_tinystories_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# compact baby GPT
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.05

learning_rate = 1e-3
max_iters = 6000
lr_decay_iters = 6000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

iters_per_pass = 73
first_pass_corpus = 'shake'

# Uniform alpha sampling: each slot sees its corpus across the full
# alpha in [0, 1] range during its training window.
mix_distribution = 'uniform'
sample_alpha_beta_every = 1

# alt_mixed: mixed batches every iter, but pass-level alternation of
# which slot's AdamW optimizer steps. The off-slot's m_t / v_t are
# preserved exactly during its off pass.
batch_mode = 'alt_mixed'

# Gradient normalization off — alt_mixed addresses slot co-dependence
# directly without needing magnitude tricks.
gradient_normalize = False
grad_norm_floor = 0.05

save_every_passes = 7

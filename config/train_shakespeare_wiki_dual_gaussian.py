# train a miniature character-level shakespeare+wiki model with DUAL WEIGHT
# SLOTS and MIXED-BATCH gradient routing, using a Gaussian-like mix distribution.
#
# Same as train_shakespeare_wiki_dual_mixed.py but the per-iter (alpha, beta)
# sampling uses symmetric Beta(c, c) — a smooth, single-peaked, "Gaussian-like"
# distribution on [0, 1] centered at 0.5. The standard deviation is controlled
# by `mix_std` (we solve for c internally). Smaller std => mass concentrates
# around 0.5, larger std => approaches Uniform[0, 1].
#
# Why this matters: with Uniform mixing the slots drift far apart along the
# range (alpha=0 and alpha=1 corners both get trained heavily), and the midpoint
# (where evaluation happens) suffers from destructive averaging. With a tight
# Gaussian-like distribution we train MOSTLY near (0.5, 0.5), letting the slots
# specialize only modestly. The hope: midpoint val drops because the slots stay
# coherent at the midpoint, while still gaining a small per-source signal.
#
# `mix_std` reference points (for symmetric Beta on [0,1]):
#   mix_std = 0.289  -> c = 1     (Uniform[0,1], NO concentration)
#   mix_std = 0.150  -> c ~ 5
#   mix_std = 0.100  -> c ~ 12    (default for this config — modest concentration)
#   mix_std = 0.050  -> c ~ 50
#   mix_std = 0.030  -> c ~ 138
#   mix_std = 0.010  -> c ~ 1250  (very tight)

out_dir = 'out-shakespeare-wiki-dual-gaussian'
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

iters_per_pass = 73
first_pass_corpus = 'shake'

# THE NEW BIT: Gaussian-like sampling on alpha via symmetric Beta(c, c). We
# specify std directly and let the trainer solve for c. Default of 0.10 is
# "modest concentration around 0.5" — slots will specialize slowly, midpoint
# stays well-trained.
mix_distribution = 'symmetric_beta'
mix_std = 0.10
sample_alpha_beta_every = 1

batch_mode = 'mixed'

# Save cadence (in passes). Same default as the alternating run.
save_every_passes = 7

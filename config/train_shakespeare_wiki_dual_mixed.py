# train a miniature character-level shakespeare+wiki model with DUAL WEIGHT
# SLOTS and MIXED-BATCH per-iter gradient routing.
#
# Same model as train_shakespeare_wiki_dual.py but with batch_mode='mixed':
# every batch is half-shake + half-wiki, single forward, two backward calls
# route gradients to W_s and W_w separately via Tensor.backward(inputs=...).
# No per-pass corpus drift, no oscillation in the loss curve. ~1.5x compute
# per iter (1 forward + 2 backwards instead of 1+1) in exchange for cleaner
# per-source gradient accounting and smoother convergence.

out_dir = 'out-shakespeare-wiki-dual-mixed'
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

# Pass alternation is structurally irrelevant in mixed mode (every batch
# already contains both corpora), but iters_per_pass still controls save
# cadence boundaries — keep the same value as the alternating run for an
# apples-to-apples comparison.
iters_per_pass = 73
first_pass_corpus = 'shake'

# dual-slot mixing — alpha + beta = 1.
#
# Default is now 'uniform' (α ~ U[0, 1]). The original Beta(0.5, 0.5) choice
# placed too much mass at α=0 and α=1, which made the per-iter gradient
# magnitude on each slot swing wildly (near-zero on the "off" corner, full on
# the "on" corner). AdamW's v_t (second moment) calibrated to the spikes and
# damped the in-between updates, causing the optimizer to plateau. Uniform
# keeps the gradient magnitudes roughly consistent across iters so Adam can
# adapt cleanly. Set mix_distribution='arcsine' (or the legacy 'beta_half'
# alias) to recover the original corner-emphasizing distribution for ablations.
mix_distribution = 'uniform'
sample_alpha_beta_every = 1

# THE NEW BIT vs train_shakespeare_wiki_dual.py: every batch is half-shake +
# half-wiki, two backward calls route gradients per slot via
# Tensor.backward(inputs=...).
batch_mode = 'mixed'

# Save cadence (in passes). Same default as the alternating run.
save_every_passes = 7

# on macbook also add
# device = 'cpu'
# compile = False

# alt_mixed Uniform with the "give it everything" upgrades:
#   * larger model (n_layer=8, n_embd=512, ~50M params → ~25M per slot)
#   * longer training (24000 iters, 2x the previous runs)
#   * lower dropout (0.05 — alt_mixed already has alpha-stochasticity)
#   * lower lr (8e-4, conventional for ~50M-param models)
#   * effective batch 128 via gradient_accumulation_steps=2 (smoother grads)
#
# Goal: close the gap to per-corpus single-model baselines (e.g., nanoGPT
# char at ~10.65M on shake-only gets val~1.48). The current 21M dual at
# alt_mixed lands at wiki val ~1.75-1.85; this should bring it to ~1.5-1.6.
#
# Trained on shake+wiki char with the alt_mixed scheme (two-optimizer
# pass-level alternation + mixed-batch forwards + Uniform alpha sampling).

out_dir = 'out-shakespeare-wiki-dual-xl'
eval_interval = 250
eval_iters = 200
log_interval = 20

dataset = 'shakespeare_wiki_char'
gradient_accumulation_steps = 2   # effective batch 128
batch_size = 64
block_size = 256

# BIGGER GPT
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.05    # was 0.2 — alt_mixed already regularizes via alpha noise

learning_rate = 8e-4   # was 1e-3; lower for the bigger model
max_iters = 24000      # 2x previous
lr_decay_iters = 24000
min_lr = 8e-5
beta2 = 0.99

warmup_iters = 200     # 2x previous since training is longer

iters_per_pass = 73
first_pass_corpus = 'shake'

mix_distribution = 'uniform'
sample_alpha_beta_every = 1

batch_mode = 'alt_mixed'

gradient_normalize = False
grad_norm_floor = 0.05

save_every_passes = 14   # was 7; halved snapshot frequency since this run is 2x longer

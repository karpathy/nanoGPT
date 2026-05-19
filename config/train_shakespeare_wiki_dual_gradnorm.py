# train a miniature character-level shakespeare+wiki model with DUAL WEIGHT
# SLOTS and MIXED-BATCH gradient routing + GRADIENT NORMALIZATION.
#
# The diagnostic experiment: mixed-batch runs (Uniform, Gaussian) produced
# similar val numbers to the alternating run BUT generate gibberish at
# alpha=0 and alpha=1. The hypothesis is that per-iter alpha-scaling on the
# chain-rule gradient (dL/dW_s = alpha * dL/dW) gives Adam's v_t a noisy
# magnitude signal that varies with alpha rather than with the loss
# landscape, so the slots never converge to standalone-coherent models.
#
# This config enables gradient_normalize=True. The mixed-batch backward
# divides loss_s by max(alpha, 0.05) and loss_w by max(beta, 0.05) before
# the backward call, exactly cancelling the chain-rule alpha-factor and
# delivering a unit-magnitude gradient to each slot regardless of the
# per-iter mixing weight.
#
# Hypothesis: if gradient normalization is the bottleneck, this run should
# produce readable text at alpha=0 and alpha=1 (matching alternating
# Beta_half) while keeping the smooth slider behavior of mixed-batch mode.
# If it doesn't, the alternating-mode advantage is structural and not just
# an optimization-hygiene issue.

out_dir = 'out-shakespeare-wiki-dual-gradnorm'
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

# Use Uniform alpha sampling. With gradient_normalize=True the chain-rule
# alpha-scaling no longer matters for Adam's v_t, so Uniform's full-range
# corner coverage should actually train the slots well at all alpha.
mix_distribution = 'uniform'
sample_alpha_beta_every = 1

batch_mode = 'mixed'

# THE NEW BIT: cancel chain-rule alpha-scaling before backward.
gradient_normalize = True
grad_norm_floor = 0.05

save_every_passes = 7

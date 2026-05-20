# alternating Beta(0.5,0.5) on shakespeare+TinyStories char, MEDIUM size.
#
# Companion to train_shakespeare_tinystories_dual_alt_mixed_medium.py — same
# model size and iter budget, but the original alternating batch_mode (pure
# per-corpus passes with masked-gradient slot routing) instead of alt_mixed.
# We confirmed on the wiki experiments that alternating Beta_half and
# alt_mixed Uniform were tied within noise at the compact scale, so we want
# to rerun the comparison on tinystories.
#
# Diff from alt_mixed_medium:
#   * batch_mode = 'alternating' (default; pure-per-corpus passes + mask-grads)
#   * mix_distribution = 'beta_half' (Beta(0.5, 0.5), the recipe that matched
#     alt_mixed on wiki)
#   * dropout = 0.2 (alternating doesn't have alpha-stochasticity, so it
#     needs conventional dropout to regularize)
#
# Trained on shake+tinystories char. Expected val ~1.20-1.30, ~2 hr on T4.

out_dir = 'out-shakespeare-tinystories-dual-alt-medium'
eval_interval = 250
eval_iters = 200
log_interval = 20

dataset = 'shakespeare_tinystories_char'
gradient_accumulation_steps = 2   # effective batch 128
batch_size = 64
block_size = 256

# MEDIUM GPT
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 8e-4
max_iters = 12000
lr_decay_iters = 12000
min_lr = 8e-5
beta2 = 0.99

warmup_iters = 200

iters_per_pass = 73
first_pass_corpus = 'shake'

mix_distribution = 'beta_half'
sample_alpha_beta_every = 1

batch_mode = 'alternating'

gradient_normalize = False
grad_norm_floor = 0.05

save_every_passes = 7

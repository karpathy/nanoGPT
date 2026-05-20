# alt_mixed Uniform on shakespeare+TinyStories char, MEDIUM size:
#   * n_layer=6, n_embd=512 (~25M total params / ~12.5M per slot)
#   * 2x the compact build (6M), half the wiki-XL build (50M)
#   * 12000 iters (half the wiki-XL run)
#   * dropout=0.05 (alt_mixed already regularizes via alpha-stochasticity)
#   * lr=8e-4, effective batch 128 via gradient_accumulation_steps=2
#
# Rationale: TinyStories is much simpler than Wikipedia at char-level
# (synthetic narrative, small effective vocab, simple grammar). Karpathy's
# stories260K shows 260K params already produce coherent text. The wiki-XL
# config (50M, 24k iters, ~5-6 hr on T4) would waste compute here; the
# compact 6M variant gives a clean low-end. This medium sits in the middle
# so we can show capacity scaling without an overnight run.
#
# Expected: val ~1.20-1.30 on the mixed corpus, ~2 hr on a T4.
#
# Trained on shake+tinystories char with the alt_mixed scheme (two-optimizer
# pass-level alternation + mixed-batch forwards + Uniform alpha sampling).

out_dir = 'out-shakespeare-tinystories-dual-medium'
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
dropout = 0.05    # alt_mixed already regularizes via alpha noise

learning_rate = 8e-4
max_iters = 12000
lr_decay_iters = 12000
min_lr = 8e-5
beta2 = 0.99

warmup_iters = 200

iters_per_pass = 73
first_pass_corpus = 'shake'

mix_distribution = 'uniform'
sample_alpha_beta_every = 1

batch_mode = 'alt_mixed'

gradient_normalize = False
grad_norm_floor = 0.05

save_every_passes = 7

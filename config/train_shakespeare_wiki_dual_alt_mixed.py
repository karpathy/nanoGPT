# train a miniature character-level shakespeare+wiki model with the
# ALT_MIXED scheme: mixed batches every iter (each batch is half-shake +
# half-wiki) BUT pass-level alternation of which slot's optimizer steps,
# with two separate AdamW optimizers (one per slot) so the off-slot's
# Adam state is entirely frozen during its off pass.
#
# The hypothesis: alternating mode wins on the dual-source scheme because
# each slot gets clean 73-iter training windows where Adam's m_t and v_t
# accumulate a CONSISTENT gradient direction without the other slot's
# updates polluting the joint state. Mixed-batch mode loses because both
# slots are co-updated every iter, and W_s is chasing a target that
# depends on W_w's current state (which is itself being updated). The
# norm-fix attempt was wrong about the bottleneck; the real issue is
# co-dependence, not chain-rule magnitude.
#
# alt_mixed combines:
#   * mixed-batch's "each slot sees both corpora in forward at every alpha"
#     (preserves the slider's smooth interpolation behavior at inference)
#   * alternating's "clean per-slot Adam-state training windows"
#     (each slot gets 73 consecutive iters of consistent gradient signal
#     while the other is fully frozen — not just zero-grad, but untouched)
#
# Mechanism per iter:
#   * Mixed batch (half-shake + half-wiki)
#   * Forward through W = alpha*W_s + beta*W_w (alpha sampled per iter)
#   * Compute loss_s and loss_w from the two halves
#   * If current pass trains W_s:
#       backward(loss_s, inputs=W_s_params)
#       optim_s.step()   # optim_w NOT touched, m_t/v_t preserved exactly
#   * If current pass trains W_w: symmetric
#
# Predicted outcome: midpoint val descends to ~1.85 (matching alternating's
# wiki-pass val) and corner generation at alpha=0,1 is readable (matching
# alternating). If it works, mixed-corpus exposure during training PLUS
# clean per-slot Adam windows together produce a slider that's both
# smooth (mixed-batch property) and coherent at the extremes (alternating
# property).

out_dir = 'out-shakespeare-wiki-dual-alt-mixed'
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

# Use Uniform alpha sampling. Each slot will see its corpus through forward
# passes at every alpha in [0, 1] during its training window.
mix_distribution = 'uniform'
sample_alpha_beta_every = 1

# THE NEW BIT: alt_mixed mode.
batch_mode = 'alt_mixed'

# Gradient normalization is NOT used here — the alt_mixed design addresses
# the underlying problem (slot co-dependence) without needing magnitude
# tricks.
gradient_normalize = False
grad_norm_floor = 0.05

save_every_passes = 7

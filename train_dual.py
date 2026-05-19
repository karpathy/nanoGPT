"""
train_dual.py

Per-pass alternating trainer for abcGPT with DUAL WEIGHT SLOTS.

Scheme summary (see model_dual.py for the full doc):

  Every weight in the model is held in two parallel parameter tensors,
  W_s ("shake") and W_w ("wiki"). The effective forward weight is

      W = alpha * W_s + beta * W_w        with        alpha + beta = 1

  for a single scalar alpha resampled each iter (default Beta(0.5, 0.5),
  which puts mass near 0 and 1 plus everywhere in between — the model
  has to be coherent at both corners AND on the interior). Autograd
  handles the chain rule: dL/dW_s = alpha * dL/dW, dL/dW_w = beta * dL/dW.
  After backward(), we mask the gradients on the slot NOT matching the
  batch's corpus (shake batch -> wipe W_w.grad; wiki batch -> wipe
  W_s.grad) and then step the single AdamW optimizer. The result: each
  slot is updated only by batches from its own corpus, but the model
  has to make the dual decomposition coherent across the entire
  alpha in [0, 1] manifold rather than at a single point.

This script is a parallel of train_diff_logging.py:

  - Same per-pass alternation (default iters_per_pass=73, alternates
    shake/wiki starting from --first_pass_corpus).
  - Same compressed-snapshot format (zstd + torch.save) at each pass
    boundary, named pass_PPPP_{corpus}.pt.zst (both slots saved).
  - Same resume detection: load latest pass snapshot + latest_optimizer.pt.zst.
  - iter_log.jsonl now records (alpha, beta) alongside loss/corpus.
  - NO per-iter diff stream. The dual-source decomposition is structural;
    you don't recover it post-hoc from diffs, you train it into the weights.

Usage:

    python train_dual.py config/train_shakespeare_wiki_dual.py \
        --out_dir=runs/$(date +%Y%m%d-%H%M%S)-dual

The existing train.py and train_diff_logging.py are untouched.
"""

import glob
import io
import json
import math
import os
import pickle
import re
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from model_dual import DualGPT, DualGPTConfig, set_mix, mask_grads

# -----------------------------------------------------------------------------
# defaults
# -----------------------------------------------------------------------------
out_dir = 'out-dual'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
init_from = 'scratch'

# data
dataset = 'shakespeare_wiki_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# optim
learning_rate = 1e-3
max_iters = 12000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = 12000
min_lr = 1e-4

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = ('bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
         else 'float32')
compile = False

# per-pass alternation (identical to train_diff_logging.py)
iters_per_pass = 73
first_pass_corpus = 'shake'

# dual-trainer specific
# Under the alpha + beta = 1 constraint there is one knob: alpha in [0, 1].
# mix_distribution sets the sampling law over alpha; beta is always 1 - alpha.
#   'beta_half'       -> alpha ~ Beta(0.5, 0.5)        (mass at corners + interior)
#   'uniform'         -> alpha ~ U[0, 1]               (== Beta(1, 1))
#   'arcsine'         -> equivalent to Beta(0.5, 0.5) but expressed via sin^2(pi*U/2)
#   'symmetric_beta'  -> alpha ~ Beta(c, c) where c = mix_concentration; c=1 is
#                        Uniform, c->inf collapses to delta at 0.5. The std is
#                        1/(2*sqrt(2c+1)), so the variance dial is monotone in c.
#                        Equivalent shortcut: set mix_std and leave concentration
#                        to be inferred.
mix_distribution = 'beta_half'
# Knobs for 'symmetric_beta'. Either set mix_concentration directly, or set
# mix_std (target standard deviation) and we'll solve for c. mix_std wins if
# both are set. Reference points for std:
#   c=1     -> std 0.2887  (Uniform)
#   c=2     -> std 0.2236
#   c=5     -> std 0.1508
#   c=10    -> std 0.1091
#   c=50    -> std 0.0498
#   c=100   -> std 0.0353
#   c=1000  -> std 0.0112
mix_concentration = 1.0
mix_std = None  # set to a float in (0, 0.2887] to override mix_concentration
# legacy alias kept for backwards-compat with older configs that set
# mix_sampling explicitly. If both are set, mix_distribution wins.
mix_sampling = 'beta_half'
sample_alpha_beta_every = 1      # resample alpha every N iters; 1 = every iter

# Gradient normalization for mixed-batch mode. When True, before the per-slot
# backward calls we divide loss_s by max(alpha, grad_norm_floor) and loss_w by
# max(beta, grad_norm_floor). This cancels out the chain-rule alpha-scaling
# (since W = alpha*W_s + beta*W_w means dL/dW_s = alpha * dL/dW), so each slot
# sees a "unit-magnitude" gradient regardless of the per-iter mixing weight.
# Without this, Adam's v_t gets calibrated to the per-iter alpha-variance, not
# to the loss-landscape geometry, which appears to be why mixed-batch mode's
# slot extremes (alpha=0, alpha=1) produce gibberish even when midpoint val
# looks reasonable.
gradient_normalize = False
grad_norm_floor = 0.05

# checkpointing cadence: write latest.pt.zst at every Nth pass boundary plus at
# max_iters. Default 7 ≈ every ~5 min on T4 (each pass ~40 s). Set to 1 for
# every-pass saves (only useful if you really want sub-minute resume granularity
# at the cost of ~5x more Drive I/O), or set high (e.g., 999) for a single
# end-of-run save.
save_every_passes = 7

# In 'mixed' batch mode (see below) we don't have meaningful pass boundaries to
# anchor saves on, so we save every N iters instead. Default 500 ≈ every ~7-8 min
# at T4 + fused AdamW. Final snapshot still lands at iter == max_iters.
save_every_iters = 500

# batch_mode controls whether shake/wiki batches alternate at pass granularity
# (the original scheme, where each batch is one corpus and gradients are routed
# by masking after backward) or whether each batch contains examples from BOTH
# corpora (the mixed scheme, where two backward calls each route to one slot
# via PyTorch's `Tensor.backward(inputs=...)` argument).
#
#   'alternating' -> the trainer alternates wiki/shake passes; within a pass
#                    every batch is from one corpus; mask_grads zeros the
#                    off-corpus slot after backward.
#   'mixed'       -> every batch is half-shake + half-wiki, single forward,
#                    two backward calls (one per half) each routed to its slot
#                    via inputs=W_X_params. No mask_grads. No per-pass corpus
#                    drift, no oscillation in the loss curve.
#
# The mixed scheme costs ~1.5x per iter (1 forward + 2 backwards vs 1+1) but
# converges more smoothly because every step's gradient reflects both corpora
# at the current (alpha, beta), not the last 73 iters of one corpus.
batch_mode = 'alternating'

# mini-run verification mode — runs the 4-invariant check on CPU and exits.
# Used to confirm gradient routing and byte-identical init before kicking off
# a full training run. Set --verify_dual=True at the command line.
verify_dual = False

zstd_level = 3
seed = 1337

# allowed config keys (for configurator.py compat)
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())

config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# zstd serialization (mirrors train_diff_logging.py)
# -----------------------------------------------------------------------------
try:
    import zstandard as zstd
    _HAVE_ZSTD = True
except ImportError:
    _HAVE_ZSTD = False
    print("WARNING: zstandard not installed; snapshots will be stored uncompressed. "
          "Run `pip install zstandard`.")


def _compress(buf: bytes) -> bytes:
    if not _HAVE_ZSTD:
        return buf
    return zstd.ZstdCompressor(level=zstd_level).compress(buf)


def _decompress(buf: bytes) -> bytes:
    if not _HAVE_ZSTD:
        return buf
    return zstd.ZstdDecompressor().decompress(buf)


def _state_dict_cpu_fp32(model: torch.nn.Module) -> dict:
    return {k: v.detach().to(dtype=torch.float32, device='cpu').clone()
            for k, v in model.state_dict().items()}


def _save_snapshot(sd: dict, path: str) -> int:
    buf = io.BytesIO()
    torch.save({'kind': 'snapshot', 'state_dict': sd}, buf)
    blob = _compress(buf.getvalue())
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)


# -----------------------------------------------------------------------------
# (alpha, beta) sampling. We sample a single scalar alpha in [0, 1] per iter
# and set beta = 1 - alpha. The default is Beta(0.5, 0.5) (a.k.a. the arcsine
# distribution), which puts mass at the corners (alpha=0 and alpha=1) and
# everywhere in between — the model has to be coherent at both endpoints
# AND on the interior. Modular so the law can be swapped later.
# -----------------------------------------------------------------------------
_rng = np.random.default_rng(seed + 9991)


def _std_to_concentration(target_std: float) -> float:
    """Invert Beta(c, c) variance formula: std = 1/(2*sqrt(2c+1)).
    Solving for c: c = ((1/(2*std))^2 - 1) / 2."""
    if target_std <= 0:
        raise ValueError(f"mix_std must be > 0, got {target_std}")
    max_std = 1.0 / math.sqrt(12.0)  # Uniform[0,1]'s std
    if target_std > max_std + 1e-9:
        raise ValueError(
            f"mix_std={target_std} exceeds the Uniform[0,1] std of {max_std:.6f}; "
            f"Beta(c,c) with c>=1 cannot have higher variance than Uniform.")
    return ((1.0 / (2.0 * target_std)) ** 2 - 1.0) / 2.0


def _sample_alpha_beta(strategy: str, concentration: float = 1.0):
    """Sample (alpha, beta) with alpha + beta == 1.

      'beta_half'      -> alpha ~ Beta(0.5, 0.5)
      'arcsine'        -> alpha = sin^2(pi/2 * U[0,1])  (same law as Beta(0.5, 0.5))
      'uniform'        -> alpha ~ U[0, 1]
      'symmetric_beta' -> alpha ~ Beta(c, c) with c = `concentration`
                          (c=1 is Uniform; c->inf collapses to alpha=0.5)
      'fixed_half'     -> alpha = 0.5  (degenerate, for debugging)
    """
    if strategy == 'beta_half':
        a = float(_rng.beta(0.5, 0.5))
    elif strategy == 'arcsine':
        u = float(_rng.uniform(0.0, 1.0))
        a = math.sin(math.pi * u / 2.0) ** 2
    elif strategy == 'uniform':
        a = float(_rng.uniform(0.0, 1.0))
    elif strategy == 'symmetric_beta':
        if concentration <= 0:
            raise ValueError(
                f"mix_concentration must be > 0, got {concentration}")
        a = float(_rng.beta(concentration, concentration))
    elif strategy == 'fixed_half':
        a = 0.5
    else:
        raise ValueError(f"unknown mix_distribution: {strategy!r}")
    return a, 1.0 - a


# -----------------------------------------------------------------------------
# data loaders (copied verbatim from train_diff_logging.py: split train.bin
# at the SEPARATOR boundary into shake / wiki halves)
# -----------------------------------------------------------------------------
def _find_separator_index(train_ids: np.ndarray, sep_ids: np.ndarray) -> int:
    n = len(sep_ids)
    if n == 0 or n > len(train_ids):
        return -1
    first = sep_ids[0]
    candidates = np.where(train_ids[: len(train_ids) - n + 1] == first)[0]
    for c in candidates:
        if np.array_equal(train_ids[c:c + n], sep_ids):
            return int(c)
    return -1


def _build_split_loaders(data_dir, block_size, batch_size, device, device_type, verbose=True):
    train_path = os.path.join(data_dir, 'train.bin')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    SEPARATOR = "\n\n===\n\n"
    try:
        sep_ids = np.array([stoi[c] for c in SEPARATOR], dtype=np.uint16)
    except KeyError as e:
        raise RuntimeError(f"separator char {e!r} not in vocab; was prepare.py run?")

    train_ids = np.memmap(train_path, dtype=np.uint16, mode='r')
    sep_idx = _find_separator_index(np.asarray(train_ids), sep_ids)
    if sep_idx < 0:
        raise RuntimeError("could not locate SEPARATOR in train.bin; rerun prepare.py")

    shake_end = sep_idx
    wiki_start = sep_idx + len(sep_ids)
    shake_len = shake_end
    wiki_len = len(train_ids) - wiki_start
    if verbose:
        print(f"split: shake={shake_len:,} chars [0:{shake_end}], "
              f"wiki={wiki_len:,} chars [{wiki_start}:{len(train_ids)}]")
    if shake_len <= block_size + 1 or wiki_len <= block_size + 1:
        raise RuntimeError(f"one of the halves is shorter than block_size+1")

    def _make_loader(start, end, name):
        def loader():
            data = np.memmap(train_path, dtype=np.uint16, mode='r')
            length = end - start
            ix = torch.randint(length - block_size, (batch_size,)) + start
            x = torch.stack([torch.from_numpy(
                data[int(i):int(i) + block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(
                data[int(i) + 1:int(i) + 1 + block_size].astype(np.int64)) for i in ix])
            if device_type == 'cuda':
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device); y = y.to(device)
            return x, y
        loader.name = name
        return loader

    wiki = _make_loader(wiki_start, len(train_ids), 'wiki')
    shake = _make_loader(0, shake_end, 'shake')
    return wiki, shake


def _build_val_loader(data_dir, block_size, batch_size, device, device_type):
    val_path = os.path.join(data_dir, 'val.bin')

    def loader():
        data = np.memmap(val_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(
            data[int(i):int(i) + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            data[int(i) + 1:int(i) + 1 + block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device); y = y.to(device)
        return x, y
    return loader


def _build_per_corpus_val_loaders(data_dir, block_size, batch_size, device, device_type,
                                   shake_val_fraction=0.1):
    """Build separate shake-val and wiki-val loaders.

    The shake-val loader samples random windows from the last `shake_val_fraction`
    of shake's range in train.bin. The wiki-val loader uses val.bin (which is the
    last 10% of the combined train+val stream — under prepare.py's shake-then-wiki
    concatenation, this is exclusively wiki tokens).

    Caveat: training currently does NOT exclude the shake-val range, so there is
    some leakage in shake-val for in-flight runs. The wiki-val region is properly
    disjoint from training because it lives in val.bin, not train.bin.
    """
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    SEPARATOR = "\n\n===\n\n"
    sep_ids = np.array([stoi[c] for c in SEPARATOR], dtype=np.uint16)
    train_ids = np.memmap(train_path, dtype=np.uint16, mode='r')
    sep_idx = _find_separator_index(np.asarray(train_ids), sep_ids)
    if sep_idx < 0:
        raise RuntimeError("could not locate SEPARATOR in train.bin")

    shake_end = sep_idx
    shake_val_size = int(shake_end * shake_val_fraction)
    shake_val_start = shake_end - shake_val_size
    shake_val_end = shake_end

    def shake_val():
        data = np.memmap(train_path, dtype=np.uint16, mode='r')
        length = shake_val_end - shake_val_start
        ix = torch.randint(length - block_size, (batch_size,)) + shake_val_start
        x = torch.stack([torch.from_numpy(
            data[int(i):int(i) + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            data[int(i) + 1:int(i) + 1 + block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device); y = y.to(device)
        return x, y

    def wiki_val():
        data = np.memmap(val_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(
            data[int(i):int(i) + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            data[int(i) + 1:int(i) + 1 + block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device); y = y.to(device)
        return x, y

    return shake_val, wiki_val


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if first_pass_corpus not in ('shake', 'wiki'):
    raise ValueError(f"first_pass_corpus must be 'shake' or 'wiki', got {first_pass_corpus!r}")
if max_iters % iters_per_pass != 0:
    print(f"WARNING: max_iters ({max_iters}) is not a multiple of iters_per_pass "
          f"({iters_per_pass}); final partial pass kept.")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]
ctx = (nullcontext() if device_type == 'cpu'
       else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

wiki_batch, shake_batch = _build_split_loaders(
    data_dir, block_size, batch_size, device, device_type)
val_batch = _build_val_loader(data_dir, block_size, batch_size, device, device_type)
shake_val_batch, wiki_val_batch = _build_per_corpus_val_loaders(
    data_dir, block_size, batch_size, device, device_type)


# Mixed-mode loaders: half-size batches from each corpus, concatenated per iter.
# verbose=False because the full-size build above already printed the split.
mixed_half = batch_size // 2
wiki_half_batch, shake_half_batch = _build_split_loaders(
    data_dir, block_size, mixed_half, device, device_type, verbose=False)


def _corpus_for_pass(pass_idx: int) -> str:
    if pass_idx % 2 == 0:
        return first_pass_corpus
    return 'wiki' if first_pass_corpus == 'shake' else 'shake'


def get_train_batch(corpus: str):
    if corpus == 'wiki':
        return wiki_batch()
    if corpus == 'shake':
        return shake_batch()


def get_mixed_batch():
    """Returns (X, Y, half) where the first `half` rows are shake examples
    and the remaining `batch_size - half` rows are wiki. Single concatenated
    batch suitable for one forward pass."""
    Xs, Ys = shake_half_batch()
    Xw, Yw = wiki_half_batch()
    return torch.cat([Xs, Xw], dim=0), torch.cat([Ys, Yw], dim=0), Xs.shape[0]


def _get_slot_params(raw_model, slot):
    """Enumerate all W_<slot> and b_<slot> parameters across the dual model.
    `slot` is 's' or 'w'."""
    params = []
    for m in raw_model.modules():
        for attr_name in (f'W_{slot}', f'b_{slot}'):
            if hasattr(m, attr_name):
                p = getattr(m, attr_name)
                if isinstance(p, torch.nn.Parameter):
                    params.append(p)
    return params
    raise ValueError(f"unknown corpus: {corpus!r}")


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, bias=bias, vocab_size=None,
                  dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new DualGPT model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = DualGPTConfig(**model_args)
    model = DualGPT(gptconf)
else:
    raise NotImplementedError("train_dual.py only supports init_from='scratch'")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate,
                                       (beta1, beta2), device_type)

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)


# Resolve the active sampling law early so verify_dual and the main loop can
# both refer to it. mix_distribution wins over the legacy mix_sampling alias;
# if mix_distribution is at its default and the user set mix_sampling to one
# of the supported values, honor that instead.
_active_mix = mix_distribution
if _active_mix == 'beta_half' and mix_sampling != 'beta_half' and \
        mix_sampling in ('beta_half', 'arcsine', 'uniform', 'fixed_half',
                         'symmetric_beta'):
    _active_mix = mix_sampling

# For symmetric_beta, resolve the concentration: mix_std (if set) overrides
# mix_concentration. Surface the final std so it shows up in run_meta.json.
_active_concentration = float(mix_concentration)
if _active_mix == 'symmetric_beta':
    if mix_std is not None:
        _active_concentration = _std_to_concentration(float(mix_std))
    _active_std = 1.0 / (2.0 * math.sqrt(2.0 * _active_concentration + 1.0))
    print(f"mix_distribution=symmetric_beta  concentration={_active_concentration:.4f}  "
          f"std={_active_std:.4f}")
else:
    _active_std = None


# -----------------------------------------------------------------------------
# verify_dual: 6-iter CPU mini-run that asserts the four invariants of the
# dual-slot scheme. Run as: `python train_dual.py --verify_dual=True`.
#
# Invariants:
#   (a) Before any training: W_s.data == W_w.data (byte-identical at iter 0).
#   (b) During a SHAKE iter, immediately before optimizer.step():
#         W_s.grad is not None and != 0, W_w.grad is None  (wiki slot frozen).
#   (c) During a WIKI iter, symmetric: W_w.grad != 0, W_s.grad is None.
#   (d) After running both passes: W_s and W_w have DIVERGED (no longer
#       byte-identical) on at least the parameters they were both updated on.
# -----------------------------------------------------------------------------
if verify_dual:
    print("\n[verify_dual] running 6-iter CPU mini-run...")
    from model_dual import (DualLinear as _DL, DualEmbedding as _DE,
                            DualLayerNorm as _DLN)
    _DUALS = (_DL, _DE, _DLN)

    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    # Pick a representative DualLinear to inspect grads on. lm_head is a good
    # choice — it's a DualLinear with no bias, ties to wte (so b_s is None for
    # this one). For bias check, pick the first DualLinear with a bias.
    probe_no_bias = None
    probe_with_bias = None
    for nm, mm in raw.named_modules():
        if isinstance(mm, _DL):
            if mm.b_s is None and probe_no_bias is None:
                probe_no_bias = (nm, mm)
            if mm.b_s is not None and probe_with_bias is None:
                probe_with_bias = (nm, mm)
    probe_name, probe = probe_with_bias if probe_with_bias is not None else probe_no_bias
    print(f"[verify_dual] probe param: {probe_name}")

    # (a) byte-identical at iter 0
    a_pass = True
    for nm, mm in raw.named_modules():
        if isinstance(mm, _DUALS):
            if not torch.equal(mm.W_s.data, mm.W_w.data):
                a_pass = False
                print(f"  FAIL (a): W_s != W_w at {nm}")
                break
            if getattr(mm, 'b_w', None) is not None and not torch.equal(mm.b_s.data, mm.b_w.data):
                a_pass = False
                print(f"  FAIL (a): b_s != b_w at {nm}")
                break
    print(f"  (a) W_s == W_w at iter 0: {'PASS' if a_pass else 'FAIL'}")

    # Stash iter-0 W_s and W_w snapshots for the divergence check.
    init_W_s = {nm: mm.W_s.data.clone() for nm, mm in raw.named_modules() if isinstance(mm, _DUALS)}
    init_W_w = {nm: mm.W_w.data.clone() for nm, mm in raw.named_modules() if isinstance(mm, _DUALS)}

    def _run_pass(corpus, n_iters):
        """Run n_iters training steps on `corpus`, and on the LAST iter capture
        the state of probe.W_s.grad and probe.W_w.grad immediately before
        optimizer.step() (i.e. after mask_grads). Returns (W_s_grad, W_w_grad,
        b_s_grad, b_w_grad) as cloned tensors / None."""
        captured = {'W_s': None, 'W_w': None, 'b_s': None, 'b_w': None}
        for k in range(n_iters):
            current_alpha, current_beta = _sample_alpha_beta(_active_mix, _active_concentration)
            set_mix(raw, current_alpha, current_beta)
            X, Y = get_train_batch(corpus)
            _, loss = raw(X, Y)
            scaler.scale(loss).backward()
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw.parameters(), grad_clip)
            mask_grads(raw, corpus)
            if k == n_iters - 1:
                for nm in ('W_s', 'W_w', 'b_s', 'b_w'):
                    p = getattr(probe, nm, None)
                    if p is None:
                        captured[nm] = None
                    else:
                        captured[nm] = (p.grad.clone() if p.grad is not None else None)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        return captured

    # (b) shake pass: W_s.grad != 0, W_w.grad is None
    shake_caps = _run_pass('shake', 2)
    b_pass = True
    if shake_caps['W_s'] is None or float(shake_caps['W_s'].abs().sum().item()) == 0.0:
        b_pass = False
        print(f"  FAIL (b): probe.W_s.grad is None or zero after shake pass")
    if shake_caps['W_w'] is not None:
        b_pass = False
        print(f"  FAIL (b): probe.W_w.grad is not None after shake pass (mask failed)")
    print(f"  (b) shake pass routes grads to W_s only: {'PASS' if b_pass else 'FAIL'}")

    # (c) wiki pass: W_w.grad != 0, W_s.grad is None
    wiki_caps = _run_pass('wiki', 2)
    c_pass = True
    if wiki_caps['W_w'] is None or float(wiki_caps['W_w'].abs().sum().item()) == 0.0:
        c_pass = False
        print(f"  FAIL (c): probe.W_w.grad is None or zero after wiki pass")
    if wiki_caps['W_s'] is not None:
        c_pass = False
        print(f"  FAIL (c): probe.W_s.grad is not None after wiki pass (mask failed)")
    print(f"  (c) wiki pass routes grads to W_w only: {'PASS' if c_pass else 'FAIL'}")

    # (d) W_s and W_w have diverged (no longer byte-identical) after both passes
    diverged_any = False
    moved_W_s = False
    moved_W_w = False
    for nm, mm in raw.named_modules():
        if isinstance(mm, _DUALS):
            if not torch.equal(mm.W_s.data, mm.W_w.data):
                diverged_any = True
            if not torch.equal(mm.W_s.data, init_W_s[nm]):
                moved_W_s = True
            if not torch.equal(mm.W_w.data, init_W_w[nm]):
                moved_W_w = True
    d_pass = diverged_any and moved_W_s and moved_W_w
    print(f"  (d) W_s and W_w have diverged: "
          f"{'PASS' if d_pass else 'FAIL'}  "
          f"(W_s moved={moved_W_s}, W_w moved={moved_W_w}, diverged_any={diverged_any})")

    all_pass = a_pass and b_pass and c_pass and d_pass
    print(f"[verify_dual] OVERALL: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


def estimate_val_loss():
    """Per-corpus val eval at THREE mixes:
      - (alpha=0.5, beta=0.5) — the balance midpoint (the "headline" number)
      - (alpha=1.0, beta=0.0) — W_s alone, evaluated on shake
      - (alpha=0.0, beta=1.0) — W_w alone, evaluated on wiki

    The corner evals are diagnostic: if the corners descend but the midpoint
    plateaus, the two slots ARE learning their own corpora and the averaging
    is what's broken (a parameterization problem). If the corners plateau
    too, the gradient flow itself is the issue.

    Note: under the prepare.py concatenation (shake + separator + wiki + 90/10
    split), val.bin is the all-wiki tail. To get a shake val signal we sample
    from the last 10% of shake's range in train.bin via shake_val_batch. There
    is some training leakage in the shake-val region for runs that didn't
    explicitly carve out a shake-val split during prepare; the wiki-val signal
    is properly disjoint because it lives in val.bin."""
    model.eval()
    s_mid_losses = torch.zeros(eval_iters)
    w_mid_losses = torch.zeros(eval_iters)
    s_corner_losses = torch.zeros(eval_iters)
    w_corner_losses = torch.zeros(eval_iters)

    # midpoint (0.5, 0.5)
    set_mix(model, 0.5, 0.5)
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = shake_val_batch()
            with ctx:
                _, loss = model(X, Y)
            s_mid_losses[k] = loss.item()
            X, Y = wiki_val_batch()
            with ctx:
                _, loss = model(X, Y)
            w_mid_losses[k] = loss.item()

    # shake corner (alpha=1, beta=0) — W_s alone on shake
    set_mix(model, 1.0, 0.0)
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = shake_val_batch()
            with ctx:
                _, loss = model(X, Y)
            s_corner_losses[k] = loss.item()

    # wiki corner (alpha=0, beta=1) — W_w alone on wiki
    set_mix(model, 0.0, 1.0)
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = wiki_val_batch()
            with ctx:
                _, loss = model(X, Y)
            w_corner_losses[k] = loss.item()

    # restore mix to whatever the trainer expects next (set to 0.5, 0.5; the
    # next iter will re-sample anyway)
    set_mix(model, 0.5, 0.5)
    model.train()

    s_mid = float(s_mid_losses.mean())
    w_mid = float(w_mid_losses.mean())
    s_corner = float(s_corner_losses.mean())
    w_corner = float(w_corner_losses.mean())
    return {
        'shake': s_mid, 'wiki': w_mid, 'mean': 0.5 * (s_mid + w_mid),
        'shake_corner': s_corner, 'wiki_corner': w_corner,
    }


def slot_divergence():
    """Sum of ||W_s - W_w||_F across all dual-slot tensor pairs, plus the
    relative divergence ||W_s - W_w||_F / ||W_s||_F. Diagnostic: if slots are
    NOT diverging across passes, the dual scheme is collapsing to a single
    model. If they ARE diverging, the slots are specializing.

    Naming in model_dual.py: paired params live as `<module>.W_s` / `<module>.W_w`
    and `<module>.b_s` / `<module>.b_w`."""
    abs_sq = 0.0
    norm_s_sq = 0.0
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, param in params.items():
            if name.endswith('.W_s'):
                w_w = params.get(name[:-4] + '.W_w')
            elif name.endswith('.b_s'):
                w_w = params.get(name[:-4] + '.b_w')
            else:
                continue
            if w_w is None:
                continue
            abs_sq += float(((param - w_w) ** 2).sum().item())
            norm_s_sq += float((param ** 2).sum().item())
    abs_norm = abs_sq ** 0.5
    rel = (abs_sq / norm_s_sq) ** 0.5 if norm_s_sq > 0 else 0.0
    return {'abs': abs_norm, 'rel': rel}


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# run metadata
# -----------------------------------------------------------------------------
total_passes = (max_iters + iters_per_pass - 1) // iters_per_pass
meta_out = {
    'config': config,
    'model_args': model_args,
    'param_count': sum(p.numel() for p in {id(p): p for p in model.parameters()}.values()),
    'have_zstd': _HAVE_ZSTD,
    'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'scheme': ('dual-slot-mixed-batch' if batch_mode == 'mixed'
               else 'dual-slot-alt-mixed' if batch_mode == 'alt_mixed'
               else 'dual-slot-per-pass-alternation'),
    'batch_mode': batch_mode,
    'total_passes': total_passes,
    'iters_per_pass': iters_per_pass,
    'first_pass_corpus': first_pass_corpus,
    'mix_distribution': mix_distribution,
    'mix_sampling': mix_sampling,  # legacy alias retained
    'active_mix': _active_mix,
    'mix_concentration': _active_concentration,
    'mix_std': _active_std,
    'sample_alpha_beta_every': sample_alpha_beta_every,
    'save_every_passes': save_every_passes,
    'gradient_normalize': gradient_normalize,
    'grad_norm_floor': grad_norm_floor,
}
with open(os.path.join(out_dir, 'run_meta.json'), 'w') as f:
    json.dump(meta_out, f, indent=2)

iter_log_path = os.path.join(out_dir, 'iter_log.jsonl')
iter_log_f = open(iter_log_path, 'a', buffering=1)

pass_log_path = os.path.join(out_dir, 'pass_log.jsonl')
pass_log_f = open(pass_log_path, 'a', buffering=1)


def _log_iter(d): iter_log_f.write(json.dumps(d) + '\n')
def _log_pass(d): pass_log_f.write(json.dumps(d) + '\n')


# -----------------------------------------------------------------------------
# resume detection
# -----------------------------------------------------------------------------
# For the dual-source scheme, per-pass snapshots are NOT the attribution
# objects — the model's {W_s, W_w} state IS the per-source decomposition.
# So we only keep a rolling latest.pt.zst (overwritten at each pass boundary,
# bundles model + optimizer + iter_num) plus a final.pt.zst at iter=max_iters.
# Total disk for a full run: ~160 MB instead of ~13 GB.
resume_from_pass_idx = 0
resume_iter_num = 0
running_total_bytes = 0
final_path = os.path.join(out_dir, 'final.pt.zst')
latest_path = os.path.join(out_dir, 'latest.pt.zst')
if os.path.exists(final_path):
    print(f"[resume] {out_dir} already contains final.pt.zst — run complete. Nothing to do.")
    sys.exit(0)
elif os.path.exists(latest_path):
    print(f"[resume] loading latest.pt.zst")
    with open(latest_path, 'rb') as f:
        payload = torch.load(io.BytesIO(_decompress(f.read())),
                             map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(payload['state_dict'])
    if 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
        print(f"[resume] restored optimizer state")
    resume_from_pass_idx = payload['pass_idx'] + 1
    resume_iter_num = payload['iter_num']
    running_total_bytes = os.path.getsize(latest_path)
    print(f"[resume] continuing from pass {resume_from_pass_idx}, iter {resume_iter_num + 1}")
else:
    # Fresh run: no init snapshot needed. At iter 0 W_s == W_w byte-identically
    # so the model is the vanilla nanoGPT init at every alpha — fully reproducible
    # from `DualGPT(cfg)` with `torch.manual_seed(seed)`. No disk cost on start.
    pass

t_run_start = time.time()
t_last_100 = t_run_start

# For mixed batch_mode we need the per-slot parameter lists once so each
# backward call can be routed via inputs=. In 'alternating' mode these are
# unused; the gradient routing happens via mask_grads after backward.
_raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
W_s_params = _get_slot_params(_raw_model, 's')
W_w_params = _get_slot_params(_raw_model, 'w')
if batch_mode not in ('alternating', 'mixed', 'alt_mixed'):
    raise ValueError(f"batch_mode must be 'alternating', 'mixed', or 'alt_mixed', got {batch_mode!r}")

# 'alt_mixed' mode: mixed batches every iter (each batch is half-shake + half-wiki)
# BUT pass-level alternation of WHICH slot gets updated, with two separate
# optimizers so the off-slot's Adam state is entirely frozen during its off
# pass (no zero-grad steps, no m_t/v_t decay).
#
# Hypothesis: combines mixed-batch's "each slot sees both corpora in forward at
# every alpha" with alternating-mode's "clean per-slot Adam-state training
# windows". If alternating's advantage was both the per-pass corpus selection
# AND the per-slot Adam isolation, this mode tests whether dropping the
# corpus-selection part (keeping mixed forwards) still works.
if batch_mode == 'alt_mixed':
    # Replicate the weight_decay grouping from configure_optimizers but split
    # by slot. Decay params: ndim >= 2 (typically W_*); no-decay: ndim < 2 (b_*).
    def _split_decay(params):
        decay, no_decay = [], []
        for p in params:
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        return decay, no_decay
    _ws_d, _ws_nd = _split_decay(W_s_params)
    _ww_d, _ww_nd = _split_decay(W_w_params)
    fused_avail = device_type == 'cuda'
    optim_s = torch.optim.AdamW(
        [{'params': _ws_d, 'weight_decay': weight_decay},
         {'params': _ws_nd, 'weight_decay': 0.0}],
        lr=learning_rate, betas=(beta1, beta2), fused=fused_avail)
    optim_w = torch.optim.AdamW(
        [{'params': _ww_d, 'weight_decay': weight_decay},
         {'params': _ww_nd, 'weight_decay': 0.0}],
        lr=learning_rate, betas=(beta1, beta2), fused=fused_avail)
    print(f"alt_mixed: two AdamW optimizers initialized "
          f"(W_s: {len(_ws_d)+len(_ws_nd)} tensors, W_w: {len(_ww_d)+len(_ww_nd)} tensors)")
else:
    optim_s = None
    optim_w = None

# main per-pass loop
iter_num = resume_iter_num
current_alpha, current_beta = _sample_alpha_beta(_active_mix, _active_concentration)
for pass_idx in range(resume_from_pass_idx, total_passes):
    # In alternating mode the pass corpus drives both batch selection and the
    # log label. In mixed mode every iter sees both corpora, so the per-pass
    # corpus label is meaningless — we report 'mixed' in logs to keep the
    # iter_log / pass_log honest about what the trainer is actually doing.
    corpus = 'mixed' if batch_mode == 'mixed' else _corpus_for_pass(pass_idx)
    start_iter = pass_idx * iters_per_pass + 1
    end_iter = min((pass_idx + 1) * iters_per_pass, max_iters)
    pass_t0 = time.time()
    pass_losses = []

    # alt_mixed: which slot trains this pass. Alternates between 's' and 'w'.
    # first_pass_corpus selects the initial slot; subsequent passes flip.
    if batch_mode == 'alt_mixed':
        alt_slot = 's' if (pass_idx % 2 == 0) == (first_pass_corpus == 'shake') else 'w'
        # Override the corpus label for logging — meaningful in alt_mixed.
        corpus = f'mixed_alt_{alt_slot}'
    else:
        alt_slot = None

    for iter_num in range(start_iter, end_iter + 1):
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        if batch_mode == 'alt_mixed':
            for pg in optim_s.param_groups: pg['lr'] = lr
            for pg in optim_w.param_groups: pg['lr'] = lr

        # Resample (alpha, beta) every sample_alpha_beta_every iters.
        if (iter_num - 1) % sample_alpha_beta_every == 0:
            current_alpha, current_beta = _sample_alpha_beta(_active_mix, _active_concentration)
        set_mix(model, current_alpha, current_beta)

        loss_shake_val = None
        loss_wiki_val = None
        if batch_mode == 'mixed':
            # Single forward over half-shake + half-wiki. Two backward calls,
            # each routed to its slot via inputs=, so the gradient accounting
            # is exact at the per-example level. No mask_grads needed.
            X, Y, half = get_mixed_batch()
            with ctx:
                logits, _ = model(X, Y)
                B, T, V = logits.shape
                per_tok = torch.nn.functional.cross_entropy(
                    logits.view(-1, V), Y.view(-1),
                    ignore_index=-1, reduction='none'
                ).view(B, T)
                per_ex = per_tok.mean(dim=1)
                loss_s = per_ex[:half].mean()
                loss_w = per_ex[half:].mean()
                loss_shake_val = float(loss_s.item())
                loss_wiki_val = float(loss_w.item())
                loss_value = 0.5 * (loss_shake_val + loss_wiki_val)
            pass_losses.append(loss_value)
            optimizer.zero_grad(set_to_none=True)
            # Gradient normalization: cancel out the chain-rule alpha-scaling
            # so each slot sees a unit-magnitude gradient regardless of the
            # current iter's (alpha, beta). Floored at grad_norm_floor to
            # prevent division-by-tiny-alpha amplifying noise. See the config
            # comment for the diagnosis.
            if gradient_normalize:
                a_scale = max(current_alpha, grad_norm_floor)
                b_scale = max(current_beta, grad_norm_floor)
                loss_s_for_grad = loss_s / a_scale
                loss_w_for_grad = loss_w / b_scale
            else:
                loss_s_for_grad = loss_s
                loss_w_for_grad = loss_w
            scaler.scale(loss_s_for_grad).backward(inputs=W_s_params, retain_graph=True)
            scaler.scale(loss_w_for_grad).backward(inputs=W_w_params)
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        elif batch_mode == 'alt_mixed':
            # Mixed batch every iter; but only one slot's optimizer steps this pass.
            # The other slot's parameters AND its Adam state are completely
            # untouched until its turn comes.
            X, Y, half = get_mixed_batch()
            with ctx:
                logits, _ = model(X, Y)
                B, T, V = logits.shape
                per_tok = torch.nn.functional.cross_entropy(
                    logits.view(-1, V), Y.view(-1),
                    ignore_index=-1, reduction='none'
                ).view(B, T)
                per_ex = per_tok.mean(dim=1)
                loss_s = per_ex[:half].mean()
                loss_w = per_ex[half:].mean()
                loss_shake_val = float(loss_s.item())
                loss_wiki_val = float(loss_w.item())
                loss_value = 0.5 * (loss_shake_val + loss_wiki_val)
            pass_losses.append(loss_value)
            if alt_slot == 's':
                optim_s.zero_grad(set_to_none=True)
                scaler.scale(loss_s).backward(inputs=W_s_params)
                if grad_clip != 0.0:
                    scaler.unscale_(optim_s)
                    torch.nn.utils.clip_grad_norm_(W_s_params, grad_clip)
                scaler.step(optim_s)
                scaler.update()
            else:  # alt_slot == 'w'
                optim_w.zero_grad(set_to_none=True)
                scaler.scale(loss_w).backward(inputs=W_w_params)
                if grad_clip != 0.0:
                    scaler.unscale_(optim_w)
                    torch.nn.utils.clip_grad_norm_(W_w_params, grad_clip)
                scaler.step(optim_w)
                scaler.update()
        else:
            # alternating: single-corpus batch, mask the off-corpus slot.
            # The single loss IS the corpus's loss; record it on the
            # corresponding field so iter_log carries the same shape as mixed.
            X, Y = get_train_batch(corpus)
            with ctx:
                _, loss = model(X, Y)
                loss_value = float(loss.item())
            pass_losses.append(loss_value)
            if corpus == 'shake':
                loss_shake_val = loss_value
            elif corpus == 'wiki':
                loss_wiki_val = loss_value
            scaler.scale(loss).backward()
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            mask_grads(_raw_model, corpus)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        snapshot_bytes = 0
        snapshot_path = None
        if iter_num == end_iter:
            is_final = (iter_num == max_iters)
            # Only save at every save_every_passes-th pass boundary (or final).
            # Each save writes ~160-240 MB to Drive; saving every pass on a fast
            # GPU spends a meaningful fraction of total wall-time on I/O.
            should_save = is_final or ((pass_idx + 1) % save_every_passes == 0)
            if should_save:
                raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                curr_sd = _state_dict_cpu_fp32(raw_model)
                snapshot_path = os.path.join(
                    out_dir, 'final.pt.zst' if is_final else 'latest.pt.zst')
                buf = io.BytesIO()
                torch.save({'kind': 'dual_checkpoint',
                            'state_dict': curr_sd,
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'pass_idx': pass_idx,
                            'corpus_at_save': corpus,
                            'config': config}, buf)
                blob = _compress(buf.getvalue())
                with open(snapshot_path, 'wb') as f:
                    f.write(blob)
                snapshot_bytes = len(blob)
                # latest.pt.zst is overwritten, so don't accumulate its size into
                # running_total_bytes; only final.pt.zst adds to disk.
                if is_final:
                    running_total_bytes += snapshot_bytes

        t1 = time.time()
        elapsed = t1 - t_run_start
        _log_iter({
            'iter': iter_num,
            'pass_idx': pass_idx,
            'corpus': corpus,
            'alpha': current_alpha,
            'beta': current_beta,
            'loss': loss_value,
            # Per-corpus losses. In mixed mode both populated; in alternating
            # mode only the one matching the pass's corpus is set, the other
            # is null. Loss curves grouped by 'loss_shake' / 'loss_wiki' can
            # be plotted across both modes uniformly.
            'loss_shake': loss_shake_val,
            'loss_wiki': loss_wiki_val,
            'lr': lr,
            'elapsed_seconds': elapsed,
        })

        if iter_num % 100 == 0:
            dt = (t1 - t_last_100) / 100
            t_last_100 = t1
            if batch_mode in ('mixed', 'alt_mixed') and loss_shake_val is not None and loss_wiki_val is not None:
                loss_field = f"loss s={loss_shake_val:.4f} w={loss_wiki_val:.4f}"
            else:
                loss_field = f"loss {loss_value:.4f}"
            print(f"[iter {iter_num:06d}] pass {pass_idx} ({corpus}) "
                  f"{loss_field} lr {lr:.4g} (a,b)=({current_alpha:.3f},{current_beta:.3f}) "
                  f"dt {dt*1000:.1f} ms disk {running_total_bytes/1e9:.2f} GB")

    pass_t1 = time.time()
    val_loss = estimate_val_loss()  # midpoint + corner evals
    divg = slot_divergence()
    train_loss_mean = float(np.mean(pass_losses)) if pass_losses else float('nan')
    _log_pass({
        'pass_idx': pass_idx,
        'corpus': corpus,
        'start_iter': start_iter,
        'end_iter': end_iter,
        'train_loss_mean': train_loss_mean,
        'val_loss_at_pass_end': val_loss['mean'],
        'val_loss_shake': val_loss['shake'],
        'val_loss_wiki': val_loss['wiki'],
        'val_loss_shake_corner': val_loss['shake_corner'],
        'val_loss_wiki_corner': val_loss['wiki_corner'],
        'slot_divergence_abs': divg['abs'],
        'slot_divergence_rel': divg['rel'],
        'snapshot_path': snapshot_path,
        'snapshot_size_bytes': snapshot_bytes,
        'elapsed_seconds': pass_t1 - pass_t0,
    })
    snapshot_field = (f"snapshot {snapshot_bytes/1e6:.2f} MB"
                      if snapshot_bytes > 0 else "snapshot --")
    print(f"[pass {pass_idx:04d}] corpus={corpus} iters {start_iter}..{end_iter} "
          f"train_mean {train_loss_mean:.4f} "
          f"mid shake={val_loss['shake']:.4f} wiki={val_loss['wiki']:.4f} "
          f"corner shake={val_loss['shake_corner']:.4f} wiki={val_loss['wiki_corner']:.4f} "
          f"||Ws-Ww||/||Ws||={divg['rel']:.4f} "
          f"{snapshot_field} pass_elapsed {pass_t1 - pass_t0:.1f}s")

    if iter_num >= max_iters:
        break

iter_log_f.close()
pass_log_f.close()
print(f"done. total bytes written: {running_total_bytes/1e9:.2f} GB in {out_dir}")

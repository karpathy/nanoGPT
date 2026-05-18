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
#   'beta_half' -> alpha ~ Beta(0.5, 0.5)   (default; mass at corners + interior)
#   'uniform'   -> alpha ~ U[0, 1]
#   'arcsine'   -> equivalent to Beta(0.5, 0.5) but expressed via sin^2(pi*U/2)
mix_distribution = 'beta_half'
# legacy alias kept for backwards-compat with older configs that set
# mix_sampling explicitly. If both are set, mix_distribution wins.
mix_sampling = 'beta_half'
sample_alpha_beta_every = 1      # resample alpha every N iters; 1 = every iter

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


def _sample_alpha_beta(strategy: str):
    """Sample (alpha, beta) with alpha + beta == 1.

      'beta_half' -> alpha ~ Beta(0.5, 0.5)            (default)
      'arcsine'   -> alpha = sin^2(pi/2 * U[0,1])      (same law as Beta(0.5, 0.5))
      'uniform'   -> alpha ~ U[0, 1]
      'fixed_half'-> alpha = 0.5                       (degenerate, for debugging)
    """
    if strategy == 'beta_half':
        a = float(_rng.beta(0.5, 0.5))
    elif strategy == 'arcsine':
        u = float(_rng.uniform(0.0, 1.0))
        a = math.sin(math.pi * u / 2.0) ** 2
    elif strategy == 'uniform':
        a = float(_rng.uniform(0.0, 1.0))
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


def _build_split_loaders(data_dir, block_size, batch_size, device, device_type):
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


def _corpus_for_pass(pass_idx: int) -> str:
    if pass_idx % 2 == 0:
        return first_pass_corpus
    return 'wiki' if first_pass_corpus == 'shake' else 'shake'


def get_train_batch(corpus: str):
    if corpus == 'wiki':
        return wiki_batch()
    if corpus == 'shake':
        return shake_batch()
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
        mix_sampling in ('beta_half', 'arcsine', 'uniform', 'fixed_half'):
    _active_mix = mix_sampling


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
            current_alpha, current_beta = _sample_alpha_beta(_active_mix)
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
    """Pass-boundary eval on val.bin at (alpha=0.5, beta=0.5) — the centroid
    of the alpha + beta = 1 line; the equally-mixed model is the natural
    reference point for the dual-slot scheme."""
    model.eval()
    losses = torch.zeros(eval_iters)
    set_mix(model, 0.5, 0.5)
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = val_batch()
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
    model.train()
    return float(losses.mean())


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
    'scheme': 'dual-slot-per-pass-alternation',
    'total_passes': total_passes,
    'iters_per_pass': iters_per_pass,
    'first_pass_corpus': first_pass_corpus,
    'mix_distribution': mix_distribution,
    'mix_sampling': mix_sampling,  # legacy alias retained
    'sample_alpha_beta_every': sample_alpha_beta_every,
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

# main per-pass loop
iter_num = resume_iter_num
current_alpha, current_beta = _sample_alpha_beta(_active_mix)
for pass_idx in range(resume_from_pass_idx, total_passes):
    corpus = _corpus_for_pass(pass_idx)
    start_iter = pass_idx * iters_per_pass + 1
    end_iter = min((pass_idx + 1) * iters_per_pass, max_iters)
    pass_t0 = time.time()
    pass_losses = []

    for iter_num in range(start_iter, end_iter + 1):
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Resample (alpha, beta) every sample_alpha_beta_every iters.
        if (iter_num - 1) % sample_alpha_beta_every == 0:
            current_alpha, current_beta = _sample_alpha_beta(_active_mix)
        set_mix(model, current_alpha, current_beta)

        X, Y = get_train_batch(corpus)
        with ctx:
            _, loss = model(X, Y)
            loss_value = float(loss.item())
        pass_losses.append(loss_value)
        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Gradient routing: null out the slot that does NOT match the corpus,
        # AFTER backward() and BEFORE optimizer.step(). This is the heart of
        # the dual-source scheme.
        mask_grads(model._orig_mod if hasattr(model, '_orig_mod') else model, corpus)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        snapshot_bytes = 0
        snapshot_path = None
        if iter_num == end_iter:
            # Save at every pass boundary, but to a single rolling file
            # (latest.pt.zst) that's overwritten each time. At iter=max_iters
            # also write a separate final.pt.zst. Combined payload: model
            # state_dict + optimizer state + iter/pass pointer + config.
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            curr_sd = _state_dict_cpu_fp32(raw_model)
            is_final = (iter_num == max_iters)
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
            'lr': lr,
            'elapsed_seconds': elapsed,
        })

        if iter_num % 100 == 0:
            dt = (t1 - t_last_100) / 100
            t_last_100 = t1
            print(f"[iter {iter_num:06d}] pass {pass_idx} ({corpus}) "
                  f"loss {loss_value:.4f} lr {lr:.4g} (a,b)=({current_alpha:.3f},{current_beta:.3f}) "
                  f"dt {dt*1000:.1f} ms disk {running_total_bytes/1e9:.2f} GB")

    pass_t1 = time.time()
    val_loss = estimate_val_loss()
    train_loss_mean = float(np.mean(pass_losses)) if pass_losses else float('nan')
    _log_pass({
        'pass_idx': pass_idx,
        'corpus': corpus,
        'start_iter': start_iter,
        'end_iter': end_iter,
        'train_loss_mean': train_loss_mean,
        'val_loss_at_pass_end': val_loss,
        'snapshot_path': snapshot_path,
        'snapshot_size_bytes': snapshot_bytes,
        'elapsed_seconds': pass_t1 - pass_t0,
    })
    print(f"[pass {pass_idx:04d}] corpus={corpus} iters {start_iter}..{end_iter} "
          f"train_mean {train_loss_mean:.4f} val@(.5,.5) {val_loss:.4f} "
          f"snapshot {snapshot_bytes/1e6:.2f} MB pass_elapsed {pass_t1 - pass_t0:.1f}s")

    if iter_num >= max_iters:
        break

iter_log_f.close()
pass_log_f.close()
print(f"done. total bytes written: {running_total_bytes/1e9:.2f} GB in {out_dir}")

"""
train_diff_logging.py

Per-pass alternating trainer for abcGPT (nanoGPT fork) with per-iter
weight-diff logging as a side artifact.

What this script does that train.py does not:

  1. Splits the combined shakespeare+wiki train.bin into two halves at the
     SEPARATOR boundary so we can serve "shake-only" and "wiki-only" batches.
  2. **Per-pass alternation.** Trains in fixed-length passes of
     `iters_per_pass` iters. Within a pass, every batch is drawn from one
     corpus only. Passes alternate: by default p=0 -> shake, p=1 -> wiki,
     p=2 -> shake, ... (configurable via --first_pass_corpus). At the end
     of each pass a full fp32 snapshot is written to
     `pass_{p:04d}_{corpus}.pt.zst`.
  3. **Per-iter diffs.** In addition to the pass snapshots, after every
     optimizer.step() the script captures
        diff_k = state_dict_k - state_dict_{k-1}
     and writes it (zstd compressed). The fine-grained diff stream is a
     side artifact for "how far back can attribution reach" sweeps; the
     pass snapshots are the primary attribution objects.
  4. Writes `iter_log.jsonl` (one row per iter, with the corpus tag) and
     `pass_log.jsonl` (one row per completed pass with val loss).
  5. Runs a 5-iter mini-warmup first to report step time vs diff-capture
     overhead.

Usage:

    # full run from a config file (still uses configurator.py override style)
    python train_diff_logging.py config/train_shakespeare_wiki_char.py \
        --out_dir=runs/$(date +%Y%m%d-%H%M%S)

    # round-trip verification (no real training, CPU-only, multi-pass)
    python train_diff_logging.py --verify_roundtrip=True

The existing train.py is not touched; this is an additive script.
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

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# defaults (overridable by config file via configurator.py, or by --flag=value)
# kept compatible with train.py's defaults so configurator.py can be reused.
# -----------------------------------------------------------------------------
out_dir = 'out-diff-logging'
eval_interval = 250  # unused for cadence (eval runs at pass boundaries) but
                     # kept so configurator.py doesn't reject existing configs
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
compile = False  # diff capture is easier without torch.compile name-mangling

# diff-logging specific
save_diffs = False         # if True, emit a per-iter diff stream (in addition to per-pass snapshots). Default off: snapshots alone are the per-source attribution objects.
save_every = 1             # ignored unless save_diffs=True; write a diff every N iters
quantize_diffs = False     # if True, int8 quantize per-tensor with fp32 scale
zstd_level = 3             # zstandard compression level; 3 is the sweet spot
verify_roundtrip = False   # if True, run multi-pass sanity test and exit
seed = 1337

# per-pass scheme
iters_per_pass = 73        # iters drawn from a single corpus per pass (~1 epoch over a 1.2M-char corpus half at batch_size*block_size=16384 tokens/iter)
first_pass_corpus = 'shake'  # 'shake' or 'wiki' -- corpus used for pass index 0

# allowed config keys (for configurator.py compat)
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# -----------------------------------------------------------------------------
# CLI parsing: we keep configurator.py's pattern (positional = config file,
# --key=value overrides) but also accept a small set of named long-options.
# -----------------------------------------------------------------------------
# configurator.py manipulates globals() directly, which is what we want.
if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())

config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# zstd import (defer so the round-trip test still works in minimal envs by
# falling back to raw bytes if zstandard isn't installed)
# -----------------------------------------------------------------------------
try:
    import zstandard as zstd
    _HAVE_ZSTD = True
except ImportError:
    _HAVE_ZSTD = False
    print("WARNING: zstandard not installed; diffs will be stored uncompressed. "
          "Run `pip install zstandard` for ~3-5x size reduction.")


def _compress(buf: bytes) -> bytes:
    if not _HAVE_ZSTD:
        return buf
    return zstd.ZstdCompressor(level=zstd_level).compress(buf)


def _decompress(buf: bytes) -> bytes:
    if not _HAVE_ZSTD:
        return buf
    return zstd.ZstdDecompressor().decompress(buf)


# -----------------------------------------------------------------------------
# snapshot / diff serialization
# -----------------------------------------------------------------------------
def _state_dict_cpu_fp32(model: torch.nn.Module) -> dict:
    """Snapshot the trainable params as CPU fp32 tensors, detached + cloned."""
    return {k: v.detach().to(dtype=torch.float32, device='cpu').clone()
            for k, v in model.state_dict().items()}


def _save_snapshot(sd: dict, path: str) -> int:
    """Serialize a full fp32 state_dict with torch.save + zstd. Returns bytes written."""
    buf = io.BytesIO()
    torch.save({'kind': 'snapshot', 'state_dict': sd}, buf)
    blob = _compress(buf.getvalue())
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)


def _load_snapshot(path: str) -> dict:
    with open(path, 'rb') as f:
        blob = _decompress(f.read())
    return torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)['state_dict']


def _compute_diff(curr: dict, prev: dict) -> dict:
    """Return curr - prev tensor-wise. Assumes identical key sets and shapes."""
    out = {}
    for k in curr:
        out[k] = curr[k] - prev[k]
    return out


def _maybe_quantize(diff: dict) -> dict:
    """Per-tensor symmetric int8 quantization with fp32 scale stored alongside."""
    q = {}
    for k, v in diff.items():
        absmax = float(v.abs().max().item())
        if absmax == 0.0:
            scale = 1.0
            q_tensor = torch.zeros_like(v, dtype=torch.int8)
        else:
            scale = absmax / 127.0
            q_tensor = torch.clamp(torch.round(v / scale), -127, 127).to(torch.int8)
        q[k] = {'q': q_tensor, 'scale': scale}
    return q


def _maybe_dequantize(diff_q: dict) -> dict:
    out = {}
    for k, v in diff_q.items():
        out[k] = v['q'].to(torch.float32) * v['scale']
    return out


def _save_diff(diff: dict, path: str, quantize: bool) -> int:
    payload = {'kind': 'diff', 'quantized': bool(quantize)}
    if quantize:
        payload['diff'] = _maybe_quantize(diff)
    else:
        payload['diff'] = diff
    buf = io.BytesIO()
    torch.save(payload, buf)
    blob = _compress(buf.getvalue())
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)


def _load_diff(path: str) -> dict:
    with open(path, 'rb') as f:
        blob = _decompress(f.read())
    payload = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    if payload.get('quantized'):
        return _maybe_dequantize(payload['diff'])
    return payload['diff']


def _apply_diff(prev: dict, diff: dict) -> dict:
    return {k: prev[k] + diff[k] for k in prev}


# -----------------------------------------------------------------------------
# data loaders: partition train.bin at the separator boundary
# -----------------------------------------------------------------------------
def _find_separator_index(train_ids: np.ndarray, sep_ids: np.ndarray) -> int:
    """Locate the start index of sep_ids within train_ids. Returns -1 if absent."""
    n = len(sep_ids)
    if n == 0 or n > len(train_ids):
        return -1
    # quick search: scan candidate positions where the first byte matches
    first = sep_ids[0]
    candidates = np.where(train_ids[: len(train_ids) - n + 1] == first)[0]
    for c in candidates:
        if np.array_equal(train_ids[c:c + n], sep_ids):
            return int(c)
    return -1


def _build_split_loaders(data_dir: str, block_size: int, batch_size: int,
                         device: str, device_type: str):
    """Build (wiki_loader, shake_loader). Returns callables that yield (X, Y)."""
    train_path = os.path.join(data_dir, 'train.bin')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    SEPARATOR = "\n\n===\n\n"
    try:
        sep_ids = np.array([stoi[c] for c in SEPARATOR], dtype=np.uint16)
    except KeyError as e:
        raise RuntimeError(
            f"separator char {e!r} not in vocab; was prepare.py run?")

    train_ids = np.memmap(train_path, dtype=np.uint16, mode='r')
    sep_idx = _find_separator_index(np.asarray(train_ids), sep_ids)
    if sep_idx < 0:
        raise RuntimeError(
            "could not locate SEPARATOR token sequence in train.bin. "
            "Re-run data/shakespeare_wiki_char/prepare.py.")

    # NOTE: prepare.py writes shake + SEPARATOR + wiki, then takes the first 90%
    # of the *characters* as train. So train.bin layout is:
    #     [shake_chars ... SEPARATOR ... wiki_chars (truncated at 90% of total)]
    # Shake-half = train_ids[0 : sep_idx]
    # Wiki-half = train_ids[sep_idx + len(sep_ids) : ]
    shake_end = sep_idx
    wiki_start = sep_idx + len(sep_ids)
    shake_len = shake_end
    wiki_len = len(train_ids) - wiki_start
    print(f"split: shake={shake_len:,} chars [0:{shake_end}], "
          f"wiki={wiki_len:,} chars [{wiki_start}:{len(train_ids)}]")
    if shake_len <= block_size + 1 or wiki_len <= block_size + 1:
        raise RuntimeError(
            f"one of the halves is shorter than block_size+1={block_size+1}; "
            "cannot sample.")

    def _make_loader(start: int, end: int, name: str):
        def loader():
            # remap each batch (avoids the memmap leak; see train.py)
            data = np.memmap(train_path, dtype=np.uint16, mode='r')
            length = end - start
            ix = torch.randint(length - block_size, (batch_size,)) + start
            x = torch.stack([torch.from_numpy(
                data[int(i):int(i) + block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(
                data[int(i) + 1:int(i) + 1 + block_size].astype(np.int64))
                for i in ix])
            if device_type == 'cuda':
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)
            return x, y
        loader.name = name
        return loader

    wiki = _make_loader(wiki_start, len(train_ids), 'wiki')
    shake = _make_loader(0, shake_end, 'shake')
    return wiki, shake


def _build_val_loader(data_dir: str, block_size: int, batch_size: int,
                      device: str, device_type: str):
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
            x = x.to(device)
            y = y.to(device)
        return x, y
    return loader


# -----------------------------------------------------------------------------
# round-trip verification (no GPU, no real data needed)
#
# Simulates 3 passes * 2 iters/pass = 6 iters on a tiny GPT. Checks:
#   (a) pass snapshots saved at end of each pass match in-memory state_dict
#       within fp32 round-trip tolerance.
#   (b) applying per-iter diffs from iter k+1..n to snapshot-at-iter-k
#       reproduces snapshot-at-iter-n.
# -----------------------------------------------------------------------------
def _run_verify_roundtrip():
    print("[verify] starting multi-pass round-trip test on CPU ...")
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16,
                    bias=False, vocab_size=57, dropout=0.0)
    model = GPT(cfg).to('cpu')
    opt = model.configure_optimizers(weight_decay=0.0, learning_rate=1e-3,
                                     betas=(0.9, 0.99), device_type='cpu')

    tmpdir = '/tmp/abcgpt_diff_verify'
    os.makedirs(tmpdir, exist_ok=True)
    # clear any prior verify artifacts so file enumeration is meaningful
    for fname in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, fname))

    # initial state, before iter 1
    sd_iter = {0: _state_dict_cpu_fp32(model)}
    init_path = os.path.join(tmpdir, 'pass_init.pt.zst')
    _save_snapshot(sd_iter[0], init_path)

    n_passes = 3
    iters_per_pass_v = 2
    first_v = 'shake'
    total_iters = n_passes * iters_per_pass_v

    pass_snapshots = {}   # pass_idx -> (last_iter, path, corpus)
    diff_paths = {}       # iter -> path

    prev_sd = sd_iter[0]
    iter_num = 0
    for p in range(n_passes):
        corpus = (first_v if p % 2 == 0
                  else ('wiki' if first_v == 'shake' else 'shake'))
        start_iter = p * iters_per_pass_v + 1
        end_iter = (p + 1) * iters_per_pass_v
        for it in range(start_iter, end_iter + 1):
            iter_num = it
            X = torch.randint(0, 57, (2, 16))
            Y = torch.randint(0, 57, (2, 16))
            _, loss = model(X, Y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            curr = _state_dict_cpu_fp32(model)
            sd_iter[it] = curr
            diff = _compute_diff(curr, prev_sd)
            d_path = os.path.join(tmpdir, f'diff_{it:06d}.pt.zst')
            _save_diff(diff, d_path, quantize=False)
            diff_paths[it] = d_path
            prev_sd = curr

        # end of pass: save snapshot
        snap_path = os.path.join(tmpdir, f'pass_{p:04d}_{corpus}.pt.zst')
        _save_snapshot(sd_iter[end_iter], snap_path)
        pass_snapshots[p] = (end_iter, snap_path, corpus)

    # (a) pass snapshots round-trip
    snap_ok = True
    snap_max = 0.0
    for p, (end_iter, snap_path, corpus) in pass_snapshots.items():
        loaded = _load_snapshot(snap_path)
        target = sd_iter[end_iter]
        m = max((target[k] - loaded[k]).abs().max().item() for k in target)
        snap_max = max(snap_max, m)
        if m >= 1e-6:
            snap_ok = False
        print(f"[verify] pass {p} ({corpus}) snapshot at iter {end_iter}: "
              f"max abs err {m:.3e}")
    print(f"[verify] (a) pass snapshots match in-memory state_dict: "
          f"{'PASS' if snap_ok else 'FAIL'} (max {snap_max:.3e})")

    # (b) per-iter diffs roll up correctly: take snapshot at iter k=end of
    # pass p, apply diffs k+1..n, compare to snapshot at iter n=end of pass q>p.
    rollup_ok = True
    rollup_max = 0.0
    pass_keys = sorted(pass_snapshots.keys())
    for i in range(len(pass_keys)):
        for j in range(i + 1, len(pass_keys)):
            k_iter, k_path, _ = pass_snapshots[pass_keys[i]]
            n_iter, n_path, _ = pass_snapshots[pass_keys[j]]
            sd = _load_snapshot(k_path)
            for it in range(k_iter + 1, n_iter + 1):
                diff = _load_diff(diff_paths[it])
                sd = _apply_diff(sd, diff)
            target = _load_snapshot(n_path)
            m = max((target[k] - sd[k]).abs().max().item() for k in target)
            rollup_max = max(rollup_max, m)
            if m >= 1e-6:
                rollup_ok = False
            print(f"[verify] rollup pass {pass_keys[i]} (iter {k_iter}) -> "
                  f"pass {pass_keys[j]} (iter {n_iter}): max abs err {m:.3e}")
    print(f"[verify] (b) diffs roll up across passes: "
          f"{'PASS' if rollup_ok else 'FAIL'} (max {rollup_max:.3e})")

    # also exercise the quantized path so we don't silently regress
    diff_q_path = os.path.join(tmpdir, 'diff_q.pt.zst')
    sample_diff = _compute_diff(sd_iter[total_iters], sd_iter[total_iters - 1])
    _save_diff(sample_diff, diff_q_path, quantize=True)
    sample_dq = _load_diff(diff_q_path)
    q_max = max((sample_diff[k] - sample_dq[k]).abs().max().item()
                for k in sample_diff)
    print(f"[verify] quantized path max abs error: {q_max:.3e} "
          f"(lossy, expected nonzero)")

    overall_pass = snap_ok and rollup_ok
    print(f"[verify] OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if verify_roundtrip:
    sys.exit(_run_verify_roundtrip())


# -----------------------------------------------------------------------------
# main training loop
# -----------------------------------------------------------------------------
if first_pass_corpus not in ('shake', 'wiki'):
    raise ValueError(
        f"first_pass_corpus must be 'shake' or 'wiki', got {first_pass_corpus!r}")
if max_iters % iters_per_pass != 0:
    print(f"WARNING: max_iters ({max_iters}) is not a multiple of "
          f"iters_per_pass ({iters_per_pass}); the final partial pass will "
          f"still be saved with its actual last iter index.")

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

# data
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
    """Even passes use first_pass_corpus, odd passes use the other one."""
    if pass_idx % 2 == 0:
        return first_pass_corpus
    return 'wiki' if first_pass_corpus == 'shake' else 'shake'


def get_train_batch(corpus: str):
    """Draw a batch from the named corpus's half of train.bin."""
    if corpus == 'wiki':
        X, Y = wiki_batch()
    elif corpus == 'shake':
        X, Y = shake_batch()
    else:
        raise ValueError(f"unknown corpus: {corpus!r}")
    return X, Y


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, bias=bias, vocab_size=None,
                  dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
else:
    raise NotImplementedError(
        "train_diff_logging.py only supports init_from='scratch' for now; "
        "resuming a diff-logged run requires reconstructing the rolling prev "
        "snapshot, which isn't wired up yet.")

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


def estimate_val_loss():
    """Run eval_iters forward passes on val.bin only. Pass-boundary eval."""
    model.eval()
    losses = torch.zeros(eval_iters)
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
# write run metadata up front
# -----------------------------------------------------------------------------
total_passes = (max_iters + iters_per_pass - 1) // iters_per_pass
meta_out = {
    'config': config,
    'model_args': model_args,
    'param_count': sum(p.numel() for p in model.parameters()),
    'have_zstd': _HAVE_ZSTD,
    'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'scheme': 'per-pass-alternation',
    'total_passes': total_passes,
    'iters_per_pass': iters_per_pass,
    'first_pass_corpus': first_pass_corpus,
}
with open(os.path.join(out_dir, 'run_meta.json'), 'w') as f:
    json.dump(meta_out, f, indent=2)

iter_log_path = os.path.join(out_dir, 'iter_log.jsonl')
iter_log_f = open(iter_log_path, 'a', buffering=1)  # line-buffered

pass_log_path = os.path.join(out_dir, 'pass_log.jsonl')
pass_log_f = open(pass_log_path, 'a', buffering=1)


def _log_iter(d: dict):
    iter_log_f.write(json.dumps(d) + '\n')


def _log_pass(d: dict):
    pass_log_f.write(json.dumps(d) + '\n')


# -----------------------------------------------------------------------------
# profile-then-train: 5-iter warmup that measures step vs save overhead
# -----------------------------------------------------------------------------
def _warmup_profile():
    print("[warmup] profiling 5 iters for step vs save overhead ...")
    model.train()
    prev_sd = _state_dict_cpu_fp32(model)
    step_times, save_times, sizes = [], [], []
    warmup_corpus = _corpus_for_pass(0)
    for i in range(5):
        X, Y = get_train_batch(warmup_corpus)
        t_a = time.time()
        with ctx:
            _, loss = model(X, Y)
        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        t_b = time.time()
        curr_sd = _state_dict_cpu_fp32(model)
        diff = _compute_diff(curr_sd, prev_sd)
        # write to /dev/null-ish location to measure compress + write
        tmp_path = os.path.join(out_dir, f'_warmup_diff_{i}.pt.zst')
        nbytes = _save_diff(diff, tmp_path, quantize=quantize_diffs)
        os.remove(tmp_path)
        t_c = time.time()
        prev_sd = curr_sd
        step_times.append(t_b - t_a)
        save_times.append(t_c - t_b)
        sizes.append(nbytes)

    avg_step = sum(step_times) / len(step_times)
    avg_save = sum(save_times) / len(save_times)
    avg_size = sum(sizes) / len(sizes)
    overhead = avg_save / max(avg_step, 1e-9)
    print(f"[warmup] avg step: {avg_step*1000:.1f} ms")
    print(f"[warmup] avg diff capture+compress+write: {avg_save*1000:.1f} ms "
          f"({overhead*100:.0f}% of step)")
    print(f"[warmup] avg compressed diff size: {avg_size/1e6:.2f} MB")
    if overhead > 0.20:
        print("[warmup] WARNING: diff-capture overhead exceeds 20% of step "
              "time. Consider --save_every=N>1 (or run with diff capture off "
              "by setting --save_every to a very large number).")
    # NOTE: the warmup *did* perform 5 real optimizer steps. Returning prev_sd
    # lets the main loop diff iter 1's state against the post-warmup state.
    return prev_sd


# -----------------------------------------------------------------------------
# resume detection: if out_dir already has pass snapshots, pick up where we
# left off instead of starting over. Colab kills sessions occasionally; this
# lets the user re-mount Drive, set RUN_ID to the existing run, and continue.
# -----------------------------------------------------------------------------
resume_from_pass_idx = 0
resume_iter_num = 0
prev_sd = None
running_total_bytes = 0
existing_snaps = sorted(glob.glob(os.path.join(out_dir, 'pass_[0-9]*_*.pt.zst')))
if existing_snaps:
    latest_path = existing_snaps[-1]
    m = re.search(r'pass_(\d+)_', os.path.basename(latest_path))
    if not m:
        raise RuntimeError(f"cannot parse pass index from {latest_path}")
    latest_pass_idx = int(m.group(1))
    latest_end_iter = min((latest_pass_idx + 1) * iters_per_pass, max_iters)
    if latest_end_iter >= max_iters:
        print(f"[resume] {out_dir} already contains a completed run ({len(existing_snaps)} passes). Nothing to do.")
        sys.exit(0)
    print(f"[resume] found {len(existing_snaps)} existing pass snapshots; "
          f"latest is pass {latest_pass_idx} ending at iter {latest_end_iter}")
    print(f"[resume] loading model state from {latest_path}")
    with open(latest_path, 'rb') as f:
        payload = torch.load(io.BytesIO(_decompress(f.read())),
                             map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(payload['state_dict'])
    # restore optimizer if a rolling state was saved alongside
    opt_path = os.path.join(out_dir, 'latest_optimizer.pt.zst')
    if os.path.exists(opt_path):
        with open(opt_path, 'rb') as f:
            opt_payload = torch.load(io.BytesIO(_decompress(f.read())),
                                     map_location=device, weights_only=False)
        optimizer.load_state_dict(opt_payload['state_dict'])
        print(f"[resume] restored optimizer state from {opt_path}")
    else:
        print(f"[resume] WARNING: no latest_optimizer.pt.zst found -- "
              f"optimizer momentum re-warms; expect a small loss bump for ~10 iters")
    resume_from_pass_idx = latest_pass_idx + 1
    resume_iter_num = latest_end_iter
    prev_sd = _state_dict_cpu_fp32(raw_model)
    running_total_bytes = sum(os.path.getsize(p)
                              for p in glob.glob(os.path.join(out_dir, '*.pt.zst')))
    print(f"[resume] continuing from pass {resume_from_pass_idx}, iter {resume_iter_num + 1}, "
          f"disk total so far {running_total_bytes/1e9:.2f} GB")
else:
    # fresh run: profile + write the init snapshot
    prev_sd = _warmup_profile()
    init_path = os.path.join(out_dir, 'pass_init.pt.zst')
    init_bytes = _save_snapshot(prev_sd, init_path)
    print(f"[init] post-warmup snapshot: {init_bytes/1e6:.2f} MB -> {init_path}")
    running_total_bytes = init_bytes

t_run_start = time.time()
t_last_100 = t_run_start

# main per-pass loop
iter_num = resume_iter_num
for pass_idx in range(resume_from_pass_idx, total_passes):
    corpus = _corpus_for_pass(pass_idx)
    start_iter = pass_idx * iters_per_pass + 1
    # final pass may be short if max_iters isn't a clean multiple
    end_iter = min((pass_idx + 1) * iters_per_pass, max_iters)
    pass_t0 = time.time()
    pass_losses = []

    for iter_num in range(start_iter, end_iter + 1):
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        X, Y = get_train_batch(corpus)
        with ctx:
            _, loss = model(X, Y)
            loss_value = float(loss.item())
        pass_losses.append(loss_value)
        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # capture current state
        curr_sd = _state_dict_cpu_fp32(model)

        compressed_bytes = 0
        wrote = False
        if save_diffs and iter_num % save_every == 0:
            diff = _compute_diff(curr_sd, prev_sd)
            diff_path = os.path.join(out_dir, f'diff_{iter_num:06d}.pt.zst')
            compressed_bytes = _save_diff(diff, diff_path, quantize=quantize_diffs)
            running_total_bytes += compressed_bytes
            wrote = True

        snapshot_bytes = 0
        snapshot_path = None
        if iter_num == end_iter:
            snapshot_path = os.path.join(
                out_dir, f'pass_{pass_idx:04d}_{corpus}.pt.zst')
            snapshot_bytes = _save_snapshot(curr_sd, snapshot_path)
            running_total_bytes += snapshot_bytes
            # rolling optimizer state for resume after a Colab crash. We save
            # only the latest (overwritten each pass) so the disk overhead is
            # bounded: ~80 MB extra regardless of total passes.
            opt_path = os.path.join(out_dir, 'latest_optimizer.pt.zst')
            buf = io.BytesIO()
            torch.save({'state_dict': optimizer.state_dict(),
                        'pass_idx': pass_idx, 'iter_num': iter_num}, buf)
            with open(opt_path, 'wb') as f:
                f.write(_compress(buf.getvalue()))

        prev_sd = curr_sd

        t1 = time.time()
        elapsed = t1 - t_run_start
        _log_iter({
            'iter': iter_num,
            'pass_idx': pass_idx,
            'corpus': corpus,
            'source': corpus,  # backward-compat alias
            'loss': loss_value,
            'lr': lr,
            'elapsed_seconds': elapsed,
            'compressed_bytes': compressed_bytes,
            'wrote_diff': wrote,
        })

        if iter_num % 100 == 0:
            per_iter = running_total_bytes / max(iter_num, 1)
            projected_total = per_iter * max_iters
            dt = (t1 - t_last_100) / 100
            t_last_100 = t1
            print(f"[iter {iter_num:06d}] pass {pass_idx} ({corpus}) "
                  f"loss {loss_value:.4f} lr {lr:.4g} dt {dt*1000:.1f} ms "
                  f"disk used {running_total_bytes/1e9:.2f} GB "
                  f"proj total {projected_total/1e9:.2f} GB")

    # end of pass: val + log
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
          f"train_mean {train_loss_mean:.4f} val {val_loss:.4f} "
          f"snapshot {snapshot_bytes/1e6:.2f} MB "
          f"pass_elapsed {pass_t1 - pass_t0:.1f}s "
          f"disk_total {running_total_bytes/1e9:.2f} GB")

    if iter_num >= max_iters:
        break

iter_log_f.close()
pass_log_f.close()
print(f"done. total bytes written: {running_total_bytes/1e9:.2f} GB "
      f"in {out_dir}")

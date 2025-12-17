# MEA in nanoGPT (experimental)

This repo includes an **experimental** integration of **Matrix Exponential Attention (MEA)** as an alternative attention backend.

## What’s implemented

- `attn_type='mea'` option in `nanogpt/model.py` that replaces softmax attention with a truncated Taylor series:
  - Order `H=0`: `V`
  - Order `H=1`: `V + A V`
  - Order `H=2`: `V + A V + 1/2 A^2 V`
- Causal masking is enforced by construction (strictly lower-triangular / autoregressive).
- Two reference implementations are provided (`mea_impl`):
  - `block` (default): materializes only `chunk×chunk` score blocks `A = QKᵀ` and is usually faster/more autograd-friendly.
  - `scan`: streaming scan implementation that keeps per-token prefix states within each chunk.

## How to run

### Correctness check (streaming vs naive reference)

```bash
python mea_check.py
```

### Training smoke test (uses `torch.compile` if enabled)

```bash
python data/shakespeare_char/prepare.py

python train.py config/train_shakespeare_char.py \
  --out_dir=out-shakespeare-char-mea-smoke \
  --attn_type=mea --mea_order=2 --mea_impl=block --mea_chunk_size=256 \
  --max_iters=20 --eval_interval=20 --eval_iters=20 --log_interval=1 \
  --compile=True
```

### Attention-only benchmark (SDPA vs MEA)

Forward+backward:

```bash
python mea_bench_attn.py --T=262144 --mode=fwd_bwd --dtype=bf16 --impl=block --chunk=2048 --iters=1 --warmup=1
```

Forward-only:

```bash
python mea_bench_attn.py --T=262144 --mode=fwd --dtype=bf16 --impl=block --chunk=2048 --iters=3 --warmup=1
```

## Notes on performance

This implementation is intended as a **baseline integration** that is easy to read/modify.

- For typical context sizes (e.g. `T <= 16k`), PyTorch SDPA/FlashAttention is usually much faster.
- MEA’s scaling advantage can show up at **very long sequence lengths**, but *only once you are far enough out* that quadratic attention is costly.
- Note: because this is not a fused kernel, **training-time** memory/time characteristics depend on implementation; `mea_impl=block` is generally more autograd-friendly than `scan`, but a custom kernel/backward is still the path to best performance.

### Example results (single A100-SXM4-80GB, PyTorch 2.9.1+cu128)

Attention-only benchmark (`B=1, nh=12, hs=64, dtype=bf16, impl=block, order=2, chunk=2048, fp32_accum=False`):

| T | SDPA (fwd+bwd) | MEA (fwd+bwd) | Speedup |
|---:|---:|---:|---:|
| 65,536  | 135 ms | 87 ms  | **1.55×** |
| 131,072 | 539 ms | 224 ms | **2.41×** |
| 262,144 | 2159 ms | 647 ms | **3.34×** |

If you need MEA to be compelling at smaller `T`, the next steps are typically:

- A fused kernel (e.g. Triton) to avoid Python-level overhead.
- A custom backward / checkpointing strategy to reduce training-time saved-tensor memory.

# Tests

## Running

From the repo root:

```bash
pytest tests/ -x -q
```

`-x` stops on first failure, `-q` is quiet. Drop both for verbose output and full failure list.

Run a single file:

```bash
pytest tests/test_model.py -v
```

Run a single test:

```bash
pytest tests/test_model.py::test_loss_decreases_on_overfit_batch -v
```

## Prerequisites

- `pip install pytest torch numpy` (transformers/tiktoken not needed for the test suite itself, only for `from_pretrained` and BPE sampling — neither is exercised here)
- For `tests/test_data.py` and `tests/test_sampling.py`: the Shakespeare-char dataset must be prepared first:
  ```bash
  python data/shakespeare_char/prepare.py
  ```
  If it isn't, those tests **skip cleanly** with a message rather than fail.
- `tests/test_sampling.py` uses CUDA by default. On CPU-only machines, edit the `--device=cuda` line in that file or skip it.

## Baseline

On RTX 4090 / Python 3.13.5 / torch 2.11+cu130:

```
18 passed in ~6s
```

Breakdown:
- `test_model.py` — 14 tests, ~1s. CPU-only, no GPU needed.
- `test_data.py` — 4 tests, <0.1s. Reads from disk, no compute.
- `test_sampling.py` — 1 test, ~5s. Subprocess: trains 30 iters + samples 20 tokens.

If `test_sampling.py` is skipped (no data prepared or no CUDA), expect **17 passed, 1 skipped, ~1s**.

## What each file covers

### `test_model.py` — unit tests on `model.py`

Uses a tiny config (`n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=64`, ~30K params) so tests run in milliseconds while exercising the same code paths as a 124M model.

- **Config & instantiation**: `GPTConfig` defaults match GPT-2 124M shape
- **Forward pass shapes**: training mode returns `(B, T, vocab)` logits + scalar loss; inference mode only computes the last position's logits (the inference-time mini-optimization in `forward()`)
- **Determinism**: same seed → identical model → identical outputs
- **Boundary check**: `forward` asserts when `seq_len > block_size`
- **`generate()`**: appends exactly `max_new_tokens`, preserves prefix, crops context past `block_size` without crashing
- **`crop_block_size()` model surgery**: shrinks the position-embedding weight, model still runnable afterwards
- **`configure_optimizers()`**: 2D+ tensors get weight decay, 1D tensors (biases, layernorms) don't — the GPT-2 paper convention
- **`estimate_mfu()`**: returns a finite positive float
- **`get_num_params()`**: `non_embedding=True` subtracts `wpe` but keeps `wte` (because of weight tying)
- **Weight tying**: `wte` and `lm_head` share the same memory (paper-mandated)
- **Behavioral sanity** (`test_loss_decreases_on_overfit_batch`): 20 AdamW steps on a fixed batch must reduce loss by ≥50%. Catches "compiles but doesn't actually learn" — the failure mode that destroys ML codebases silently.

### `test_data.py` — contract tests on `data/shakespeare_char/`

Tests the on-disk format that `train.py:get_batch()` consumes. Skips cleanly if the dataset hasn't been prepared.

- `train.bin` / `val.bin` are `uint16` (matters for `np.memmap` reads)
- `meta.pkl` has `vocab_size` / `itos` / `stoi`, all internally consistent
- No token IDs in the bin files exceed `vocab_size` (would crash embedding lookup)
- `stoi` and `itos` are inverses

### `test_sampling.py` — end-to-end smoke test

One test, session-scoped fixture:
1. Subprocess-runs `train.py` with a tiny config for 30 iters, writes a checkpoint to a `tmp_path`
2. Subprocess-runs `sample.py` against that checkpoint, asks for 20 tokens
3. Asserts exit 0, no traceback, sample separator present in stdout

Subprocess (not import) because `train.py` and `sample.py` are scripts that execute everything at import time. Subprocess tests the actual user-facing surface.

The fixture is `scope="session"` so the training only runs once even if more sample-related tests are added.

## What's deliberately not covered

- **DDP / multi-GPU** — needs special hardware
- **`GPT.from_pretrained`** — requires ~500MB download from HuggingFace
- **Mixed precision (fp16/bf16) paths** — that's torch's autocast, not nanoGPT
- **Performance / throughput** — that's `bench.py`'s job, not a test

## Design notes

- All test files start with `from model import ...`. `conftest.py` puts the repo root on `sys.path` so this works regardless of `cwd`.
- No mocks. The model is small enough to instantiate for real.
- No fixtures shared across files beyond what `conftest.py` provides.

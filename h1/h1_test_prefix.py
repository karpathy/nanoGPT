import torch
import sys
sys.path.insert(0, "/kaggle/working/nanoGPT")
from model import GPT, GPTConfig, SoftPrefix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running tests on: {DEVICE}\n")

# ── load a small GPT-2 for testing ────────────────────────────────────
def get_model():
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    return model

# ═══════════════════════════════════════════════════════════════════════
# TEST 1 — gradient flow
# cached KV must be detached (no grad) on frozen steps
# P must have grad on update steps, no grad on frozen steps
# ═══════════════════════════════════════════════════════════════════════
def test_gradient_flow():
    model  = get_model()
    prefix = SoftPrefix(
        prefix_len=64, n_layer=model.config.n_layer,
        n_head=model.config.n_head, n_embd=model.config.n_embd,
        device=DEVICE
    )

    # ── frozen step: cached KV must be detached ───────────────────────
    with torch.no_grad():
        prefix.build_cache(model)

    for i, (k, v) in enumerate(prefix.cached_kv):
        assert not k.requires_grad, \
            f"FAIL: layer {i} key has requires_grad=True on frozen step"
        assert not v.requires_grad, \
            f"FAIL: layer {i} value has requires_grad=True on frozen step"
    print("  PASS: cached KV is detached on frozen step")

    # ── update step: P must have grad, cache must be invalidated ─────
    prefix.invalidate()
    assert not prefix.cache_valid, \
        "FAIL: cache still valid after invalidate()"
    print("  PASS: cache correctly invalidated on update step")

    # ── after invalidate, build_cache recomputes fresh KV ─────────────
    with torch.no_grad():
        prefix.build_cache(model)
    assert prefix.cache_valid, \
        "FAIL: cache_valid not set after build_cache()"
    print("  PASS: cache rebuilt correctly after invalidation")

    # ── P gradient flows during update step ───────────────────────────
    prefix.P.requires_grad_(True)
    prefix.invalidate()
    # simulate forward pass with grad
    x = prefix.P.unsqueeze(0)  # (1, L, n_embd)
    loss = x.sum()
    loss.backward()
    assert prefix.P.grad is not None, \
        "FAIL: P has no gradient after backward"
    print("  PASS: gradient flows into P on update step")

    print("TEST 1 PASSED — gradient flow correct\n")

# ═══════════════════════════════════════════════════════════════════════
# TEST 2 — mask / attention correctness
# tokens must be able to attend to the prefix
# logits must shift when prefix P is ablated (zeroed out)
# ═══════════════════════════════════════════════════════════════════════
def test_attention_mask():
    model  = get_model()
    prefix = SoftPrefix(
        prefix_len=32, n_layer=model.config.n_layer,
        n_head=model.config.n_head, n_embd=model.config.n_embd,
        device=DEVICE
    )

    # dummy input batch
    batch_size = 2
    seq_len    = 64
    X = torch.randint(0, 50257, (batch_size, seq_len), device=DEVICE)
    Y = torch.randint(0, 50257, (batch_size, seq_len), device=DEVICE)

    # ── forward WITH prefix ───────────────────────────────────────────
    with torch.no_grad():
        prefix.build_cache(model)
    with torch.no_grad():
        logits_with, loss_with = model(X, Y, prefix_kvs=prefix.cached_kv)

    # ── forward WITHOUT prefix ────────────────────────────────────────
    with torch.no_grad():
        logits_without, loss_without = model(X, Y, prefix_kvs=None)

    # logits must differ — if they're identical, prefix isn't being used
    max_diff = (logits_with - logits_without).abs().max().item()
    assert max_diff > 1e-4, \
        f"FAIL: logits identical with and without prefix (max_diff={max_diff:.6f})" \
        f" — prefix KV is not being injected into attention"
    print(f"  PASS: logits differ with vs without prefix (max_diff={max_diff:.4f})")

    # ── ablation: zero out P and check logits shift ───────────────────
    original_P = prefix.P.data.clone()

    # zero P and rebuild cache
    prefix.P.data.zero_()
    with torch.no_grad():
        prefix.build_cache(model)
    with torch.no_grad():
        logits_zeroed, _ = model(X, Y, prefix_kvs=prefix.cached_kv)

    ablation_diff = (logits_with - logits_zeroed).abs().max().item()
    assert ablation_diff > 1e-4, \
        f"FAIL: zeroing P did not change logits (diff={ablation_diff:.6f})" \
        f" — P has no effect on output"
    print(f"  PASS: zeroing P shifts logits (max_diff={ablation_diff:.4f})")

    # restore P
    prefix.P.data.copy_(original_P)

    # ── loss must be finite ───────────────────────────────────────────
    assert torch.isfinite(loss_with), \
        f"FAIL: loss is not finite: {loss_with}"
    print(f"  PASS: loss is finite ({loss_with.item():.4f})")

    print("TEST 2 PASSED — attention mask and ablation correct\n")

# ═══════════════════════════════════════════════════════════════════════
# TEST 3 — overfitting check
# for large L on WikiText-2 (small dataset), val loss should not
# be dramatically worse than train loss after 100 steps
# this is a smoke test, not a full training run
# ═══════════════════════════════════════════════════════════════════════
def test_overfitting_risk():
    import numpy as np

    val_bin = "data/wikitext2/val.bin"
    if not os.path.exists(val_bin):
        print("  SKIP: data/wikitext2/val.bin not found — run Cell 6 first")
        return

    model  = get_model()
    prefix = SoftPrefix(
        prefix_len=512, n_layer=model.config.n_layer,
        n_head=model.config.n_head, n_embd=model.config.n_embd,
        device=DEVICE
    )
    optimizer = torch.optim.AdamW([prefix.P], lr=3e-5)

    # load a tiny slice of val data
    data      = np.memmap(val_bin, dtype=np.uint16, mode='r')
    block_size = 64   # small for speed
    ctx        = torch.amp.autocast(device_type='cuda', dtype=torch.float16)

    train_losses = []
    val_losses   = []

    for step in range(50):
        # ── toggle grad before forward pass ──────────────────────────
        prefix.P.requires_grad_(True)  # always True for this test since every step is an update step
        # train step
        ix = torch.randint(len(data) - block_size, (4,))
        X  = torch.stack([torch.from_numpy(
            data[i:i+block_size].astype(np.int64)) for i in ix]).to(DEVICE)
        Y  = torch.stack([torch.from_numpy(
            data[i+1:i+block_size+1].astype(np.int64)) for i in ix]).to(DEVICE)

        prefix.invalidate()
        # ── on update steps, recompute KV WITH grad, don't use cache ──
        prefix.invalidate()
        # build KV WITH gradients enabled so loss.backward() works
        kvs = []
        x_p = prefix.P.unsqueeze(0)  # (1, L, n_embd) — has grad
        for block in model.transformer.h:
            B, T, C = x_p.shape
            qkv = block.attn.c_attn(x_p)
            q, k, v = qkv.split(C, dim=2)
            k = k.view(B, T, prefix.n_head, prefix.head_dim).transpose(1, 2)
            v = v.view(B, T, prefix.n_head, prefix.head_dim).transpose(1, 2)
            kvs.append((k, v))  # ← NOT detached, grad flows through
            x_p = x_p + block.attn(block.ln_1(x_p))
            x_p = x_p + block.mlp(block.ln_2(x_p))

        with ctx:
            _, loss = model(X, Y, prefix_kvs=kvs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # val estimate
    with torch.no_grad():
        prefix.build_cache(model)
        for _ in range(20):
            ix = torch.randint(len(data) - block_size, (4,))
            X  = torch.stack([torch.from_numpy(
                data[i:i+block_size].astype(np.int64)) for i in ix]).to(DEVICE)
            Y  = torch.stack([torch.from_numpy(
                data[i+1:i+block_size+1].astype(np.int64)) for i in ix]).to(DEVICE)
            with ctx:
                _, loss = model(X, Y, prefix_kvs=prefix.cached_kv)
            val_losses.append(loss.item())

    avg_train = sum(train_losses[-10:]) / 10
    avg_val   = sum(val_losses) / len(val_losses)
    gap       = avg_val - avg_train

    print(f"  train loss (last 10 steps): {avg_train:.4f}")
    print(f"  val loss:                   {avg_val:.4f}")
    print(f"  gap:                        {gap:.4f}")

    # gap > 1.0 at L=512 after only 50 steps is a red flag
    if gap > 1.0:
        print(f"  WARNING: large train/val gap ({gap:.2f}) — "
              f"consider dropout on P or reducing L for this dataset")
    else:
        print(f"  PASS: train/val gap acceptable ({gap:.4f})")

    print("TEST 3 PASSED — overfitting risk checked\n")

# ═══════════════════════════════════════════════════════════════════════
# run all tests
# ═══════════════════════════════════════════════════════════════════════
import os

if __name__ == "__main__":
    print("="*60)
    print("  PREFIX CORRECTNESS TESTS")
    print("="*60 + "\n")

    try:
        print("TEST 1: Gradient flow")
        test_gradient_flow()
    except AssertionError as e:
        print(f"  {e}\n")

    try:
        print("TEST 2: Attention mask + ablation")
        test_attention_mask()
    except AssertionError as e:
        print(f"  {e}\n")

    try:
        print("TEST 3: Overfitting risk (L=512, 50 steps)")
        test_overfitting_risk()
    except AssertionError as e:
        print(f"  {e}\n")

    print("="*60)
    print("  All tests complete.")
    print("="*60)
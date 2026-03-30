import torch
import sys
import math
import os
sys.path.insert(0, "/kaggle/working/nanoGPT")
from model import GPT, GPTConfig, SoftPrefix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running tests on: {DEVICE}\n")

def get_model():
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    return model

# ═══════════════════════════════════════════════════════════════════
# TEST 1 — prefix affects output
# ═══════════════════════════════════════════════════════════════════
def test_prefix_affects_output():
    model  = get_model()
    prefix = SoftPrefix(32, model.config.n_embd, DEVICE).to(DEVICE)

    X = torch.randint(0, 50257, (2, 64), device=DEVICE)
    Y = torch.randint(0, 50257, (2, 64), device=DEVICE)

    with torch.no_grad():
        _, loss_no_prefix   = model(X, Y, prefix=None)
        _, loss_with_prefix = model(X, Y, prefix=prefix)

    diff = abs(loss_no_prefix.item() - loss_with_prefix.item())
    assert diff > 1e-4, f"FAIL: prefix has no effect on output (diff={diff:.6f})"
    print(f"  PASS: prefix affects output "
          f"(no_prefix={loss_no_prefix.item():.4f}, "
          f"with_prefix={loss_with_prefix.item():.4f})")
    print("TEST 1 PASSED\n")

# ═══════════════════════════════════════════════════════════════════
# TEST 2 — gradient flows to P
# ═══════════════════════════════════════════════════════════════════
def test_gradient_flows_to_P():
    model  = get_model()
    prefix = SoftPrefix(32, model.config.n_embd, DEVICE).to(DEVICE)

    X = torch.randint(0, 50257, (2, 64), device=DEVICE)
    Y = torch.randint(0, 50257, (2, 64), device=DEVICE)

    _, loss = model(X, Y, prefix=prefix)
    loss.backward()

    assert prefix.P.grad is not None, "FAIL: P has no gradient after backward"
    assert prefix.P.grad.abs().max() > 0, "FAIL: P gradient is all zeros"
    print(f"  PASS: gradient flows to P "
          f"(max_grad={prefix.P.grad.abs().max().item():.6f})")
    print("TEST 2 PASSED\n")

# ═══════════════════════════════════════════════════════════════════
# TEST 3 — zeroing P changes output
# ═══════════════════════════════════════════════════════════════════
def test_zeroing_P_changes_output():
    model  = get_model()
    prefix = SoftPrefix(32, model.config.n_embd, DEVICE).to(DEVICE)

    X = torch.randint(0, 50257, (2, 64), device=DEVICE)
    Y = torch.randint(0, 50257, (2, 64), device=DEVICE)

    with torch.no_grad():
        _, loss_random = model(X, Y, prefix=prefix)
        prefix.P.data.zero_()
        _, loss_zeroed = model(X, Y, prefix=prefix)

    diff = abs(loss_random.item() - loss_zeroed.item())
    assert diff > 1e-4, f"FAIL: zeroing P has no effect (diff={diff:.6f})"
    print(f"  PASS: zeroing P changes output "
          f"(random={loss_random.item():.4f}, zeroed={loss_zeroed.item():.4f})")
    print("TEST 3 PASSED\n")

# ═══════════════════════════════════════════════════════════════════
# TEST 4 — LM weights are frozen
# ═══════════════════════════════════════════════════════════════════
def test_lm_frozen():
    model  = get_model()
    prefix = SoftPrefix(32, model.config.n_embd, DEVICE).to(DEVICE)

    for name, param in model.named_parameters():
        assert not param.requires_grad, \
            f"FAIL: LM parameter {name} has requires_grad=True"

    assert prefix.P.requires_grad, "FAIL: P should have requires_grad=True"
    print("  PASS: all LM parameters frozen")
    print("  PASS: P requires grad")
    print("TEST 4 PASSED\n")

# ═══════════════════════════════════════════════════════════════════
# TEST 5 — prefix slice correctness
# output shape must match input shape (prefix positions removed)
# ═══════════════════════════════════════════════════════════════════
def test_output_shape():
    model  = get_model()
    prefix = SoftPrefix(32, model.config.n_embd, DEVICE).to(DEVICE)

    B, T = 2, 64
    X = torch.randint(0, 50257, (B, T), device=DEVICE)

    with torch.no_grad():
        logits_no_prefix, _ = model(X, prefix=None)
        logits_with_prefix, _ = model(X, prefix=prefix)

    assert logits_no_prefix.shape == logits_with_prefix.shape, \
        f"FAIL: shape mismatch — no_prefix={logits_no_prefix.shape}, " \
        f"with_prefix={logits_with_prefix.shape}"
    print(f"  PASS: output shape consistent {logits_with_prefix.shape}")
    print("TEST 5 PASSED\n")

# ═══════════════════════════════════════════════════════════════════
# run all
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  PREFIX CORRECTNESS TESTS")
    print("=" * 60 + "\n")

    for name, fn in [
        ("TEST 1: prefix affects output",       test_prefix_affects_output),
        ("TEST 2: gradient flows to P",          test_gradient_flows_to_P),
        ("TEST 3: zeroing P changes output",     test_zeroing_P_changes_output),
        ("TEST 4: LM weights frozen",            test_lm_frozen),
        ("TEST 5: output shape correct",         test_output_shape),
    ]:
        try:
            print(name)
            fn()
        except AssertionError as e:
            print(f"  {e}\n")

    print("=" * 60)
    print("  All tests complete.")
    print("=" * 60)
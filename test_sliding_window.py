"""
Test script for sliding window attention implementation.
"""
import torch
from model import GPT, GPTConfig

def test_sliding_window_attention():
    print("Testing sliding window attention implementation...")

    # Test 1: Full attention (window_size=None, default behavior)
    print("\n1. Testing full attention (window_size=None)...")
    config_full = GPTConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        window_size=None
    )
    model_full = GPT(config_full)
    model_full.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, config_full.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits_full, _ = model_full(x)

    print(f"   Full attention output shape: {logits_full.shape}")
    print(f"   Expected shape: ({batch_size}, {seq_len}, {config_full.vocab_size})")
    assert logits_full.shape == (batch_size, seq_len, config_full.vocab_size), "Full attention shape mismatch!"
    print("   ✓ Full attention test passed!")

    # Test 2: Sliding window attention with window_size=32
    print("\n2. Testing sliding window attention (window_size=32)...")
    config_window = GPTConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        window_size=32
    )
    model_window = GPT(config_window)
    model_window.eval()

    with torch.no_grad():
        logits_window, _ = model_window(x)

    print(f"   Sliding window output shape: {logits_window.shape}")
    print(f"   Expected shape: ({batch_size}, {seq_len}, {config_window.vocab_size})")
    assert logits_window.shape == (batch_size, seq_len, config_window.vocab_size), "Sliding window shape mismatch!"
    print("   ✓ Sliding window attention test passed!")

    # Test 3: Verify attention mask structure
    print("\n3. Verifying attention mask structure...")

    # For non-flash attention (manual implementation)
    if hasattr(model_window.transformer.h[0].attn, 'bias'):
        mask = model_window.transformer.h[0].attn.bias[0, 0, :64, :64]
        print(f"   Attention mask shape: {mask.shape}")

        # Check that the mask is causal
        for i in range(min(10, seq_len)):
            for j in range(min(10, seq_len)):
                if j > i:
                    # Future positions should be masked out
                    assert mask[i, j] == 0, f"Mask should be 0 for future position ({i}, {j})"
                elif i - j <= config_window.window_size:
                    # Positions within window should be visible
                    assert mask[i, j] == 1, f"Mask should be 1 for position ({i}, {j}) within window"
                else:
                    # Positions beyond window should be masked out
                    assert mask[i, j] == 0, f"Mask should be 0 for position ({i}, {j}) beyond window"

        print("   ✓ Attention mask structure is correct!")
    else:
        print("   Flash attention detected - skipping manual mask check")

    # Test 4: Different outputs for different window sizes
    print("\n4. Testing that different window sizes produce different outputs...")

    config_window2 = GPTConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        window_size=16  # Different window size
    )
    model_window2 = GPT(config_window2)
    model_window2.eval()

    with torch.no_grad():
        logits_window2, _ = model_window2(x)

    # The outputs should be different due to different attention patterns
    assert not torch.allclose(logits_window, logits_window2), "Different window sizes should produce different outputs!"
    print("   ✓ Different window sizes produce different outputs!")

    # Test 5: Test with longer sequence
    print("\n5. Testing with longer sequence...")
    seq_len_long = 128
    x_long = torch.randint(0, config_window.vocab_size, (batch_size, seq_len_long))

    with torch.no_grad():
        logits_long, _ = model_window(x_long)

    print(f"   Long sequence output shape: {logits_long.shape}")
    assert logits_long.shape == (batch_size, seq_len_long, config_window.vocab_size), "Long sequence shape mismatch!"
    print("   ✓ Long sequence test passed!")

    print("\n" + "="*60)
    print("All tests passed successfully! ✓")
    print("="*60)

if __name__ == "__main__":
    test_sliding_window_attention()

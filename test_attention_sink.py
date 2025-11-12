"""
Test script for Attention Sink implementation

This script verifies:
1. Model initialization with attention sink
2. Forward pass works correctly
3. Attention mask is created properly
4. Model can be trained (at least for a few steps)
"""

import torch
import numpy as np
from model import GPTConfig, GPT

def test_model_initialization():
    """Test that models initialize correctly"""
    print("Test 1: Model Initialization")
    print("-" * 40)

    # Standard model
    config_standard = GPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        use_attention_sink=False
    )
    model_standard = GPT(config_standard)
    print(f"✓ Standard model initialized: {sum(p.numel() for p in model_standard.parameters()):,} parameters")

    # Attention sink model
    config_sink = GPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        use_attention_sink=True,
        sink_size=4,
        window_size=60
    )
    model_sink = GPT(config_sink)
    print(f"✓ Attention sink model initialized: {sum(p.numel() for p in model_sink.parameters()):,} parameters")

    # Check that both models have the same number of parameters
    params_standard = sum(p.numel() for p in model_standard.parameters())
    params_sink = sum(p.numel() for p in model_sink.parameters())
    assert params_standard == params_sink, "Models should have same number of parameters!"
    print(f"✓ Parameter count matches: {params_standard:,}")

    print()
    return model_standard, model_sink, config_standard, config_sink

def test_forward_pass(model_standard, model_sink):
    """Test forward pass"""
    print("Test 2: Forward Pass")
    print("-" * 40)

    batch_size = 4
    seq_len = 64
    vocab_size = 256

    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Standard model forward
    logits_standard, loss_standard = model_standard(x, y)
    print(f"✓ Standard model forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits_standard.shape}")
    print(f"  Loss: {loss_standard.item():.4f}")

    # Attention sink model forward
    logits_sink, loss_sink = model_sink(x, y)
    print(f"✓ Attention sink model forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits_sink.shape}")
    print(f"  Loss: {loss_sink.item():.4f}")

    # Check shapes
    assert logits_standard.shape == logits_sink.shape, "Output shapes should match!"
    print(f"✓ Output shapes match")

    print()
    return loss_standard, loss_sink

def test_attention_mask(config_sink):
    """Test that attention mask is created correctly"""
    print("Test 3: Attention Mask")
    print("-" * 40)

    from model import AttentionSinkCausalSelfAttention

    attn_module = AttentionSinkCausalSelfAttention(config_sink)

    # Test mask creation
    T = 32
    device = 'cpu'
    mask = attn_module._create_sink_window_mask(T, device)

    print(f"✓ Mask created with shape: {mask.shape}")
    print(f"  Sink size: {config_sink.sink_size}")
    print(f"  Window size: {config_sink.window_size}")

    # Verify mask properties
    # 1. First row should attend to all previous positions (just itself)
    assert mask[0, 0] == True, "Position 0 should attend to itself"
    print(f"✓ Position 0 attends to itself")

    # 2. Every position should attend to sink tokens
    for i in range(T):
        for j in range(min(config_sink.sink_size, T)):
            assert mask[i, j] == True, f"Position {i} should attend to sink token {j}"
    print(f"✓ All positions attend to sink tokens (0-{config_sink.sink_size-1})")

    # 3. Positions should attend within their window
    test_pos = 20
    if test_pos < T:
        window_start = max(config_sink.sink_size, test_pos - config_sink.window_size + 1)
        # Check that we attend to positions in window
        assert mask[test_pos, test_pos] == True, f"Position {test_pos} should attend to itself"
        if window_start < test_pos:
            assert mask[test_pos, window_start] == True, f"Position {test_pos} should attend to window start"
        print(f"✓ Position {test_pos} attends within window ({window_start} to {test_pos})")

    # 4. Positions should NOT attend to tokens outside sink and window
    if T > config_sink.sink_size + config_sink.window_size + 5:
        test_pos = T - 1
        # There should be some position not attended to
        outside_pos = config_sink.sink_size + 2  # Just outside sink, will be outside window for last position
        if outside_pos < test_pos - config_sink.window_size:
            assert mask[test_pos, outside_pos] == False, f"Position {test_pos} should NOT attend to {outside_pos}"
            print(f"✓ Masking works: position {test_pos} does not attend to {outside_pos}")

    # Visualize mask (small version)
    if T <= 16:
        print("\nMask visualization (1 = attend, 0 = masked):")
        print(mask.int().numpy())

    print()

def test_training_step(model_standard, model_sink):
    """Test that model can do a training step"""
    print("Test 4: Training Step")
    print("-" * 40)

    batch_size = 4
    seq_len = 64
    vocab_size = 256

    # Create dummy data
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Standard model training step
    model_standard.train()
    optimizer_standard = torch.optim.AdamW(model_standard.parameters(), lr=1e-3)

    logits, loss = model_standard(x, y)
    optimizer_standard.zero_grad()
    loss.backward()
    optimizer_standard.step()

    print(f"✓ Standard model training step completed")
    print(f"  Loss: {loss.item():.4f}")

    # Attention sink model training step
    model_sink.train()
    optimizer_sink = torch.optim.AdamW(model_sink.parameters(), lr=1e-3)

    logits, loss = model_sink(x, y)
    optimizer_sink.zero_grad()
    loss.backward()
    optimizer_sink.step()

    print(f"✓ Attention sink model training step completed")
    print(f"  Loss: {loss.item():.4f}")

    print()

def test_generation(model_standard, model_sink):
    """Test generation"""
    print("Test 5: Text Generation")
    print("-" * 40)

    # Start with a simple prompt
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # Dummy tokens

    model_standard.eval()
    model_sink.eval()

    # Generate from standard model
    output_standard = model_standard.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"✓ Standard model generation:")
    print(f"  Input length: {prompt.shape[1]}")
    print(f"  Output length: {output_standard.shape[1]}")
    print(f"  Generated tokens: {output_standard[0].tolist()}")

    # Generate from attention sink model
    output_sink = model_sink.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"✓ Attention sink model generation:")
    print(f"  Input length: {prompt.shape[1]}")
    print(f"  Output length: {output_sink.shape[1]}")
    print(f"  Generated tokens: {output_sink[0].tolist()}")

    print()

if __name__ == "__main__":
    print("="*80)
    print("ATTENTION SINK IMPLEMENTATION TEST")
    print("="*80)
    print()

    torch.manual_seed(42)

    try:
        # Run tests
        model_standard, model_sink, config_standard, config_sink = test_model_initialization()
        test_forward_pass(model_standard, model_sink)
        test_attention_mask(config_sink)
        test_training_step(model_standard, model_sink)
        test_generation(model_standard, model_sink)

        print("="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print()
        print("The attention sink implementation is working correctly.")
        print("You can now proceed to train models using the configs:")
        print("  - config/train_baseline_for_sink.py")
        print("  - config/train_attention_sink.py")

    except Exception as e:
        print("="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

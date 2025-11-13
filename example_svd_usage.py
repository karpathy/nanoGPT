#!/usr/bin/env python3
"""
Example script demonstrating how to use the SVD toggle feature in the GPT model.
This shows how to enable/disable SVD and configure different SVD parameters.
"""

import torch
from model import GPT, GPTConfig

def create_model_with_svd(use_svd=True, svd_rank=None):
    """
    Create a GPT model with SVD configuration
    
    Args:
        use_svd: Whether to enable SVD decomposition on value matrices
        svd_rank: Rank for SVD approximation (None for full rank)
    """
    config = GPTConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        use_svd=use_svd,
        svd_rank=svd_rank
    )
    
    model = GPT(config)
    return model

def test_svd_configurations():
    """Test different SVD configurations"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create some dummy input
    batch_size = 2
    seq_length = 32
    x = torch.randint(0, 50304, (batch_size, seq_length), device=device)
    
    print("Testing different SVD configurations:")
    print("=" * 50)
    
    # Test 1: No SVD (baseline)
    print("1. Baseline (No SVD)")
    model_baseline = create_model_with_svd(use_svd=False).to(device)
    model_baseline.eval()
    with torch.no_grad():
        logits_baseline, _ = model_baseline(x)
    print(f"   Output shape: {logits_baseline.shape}")
    print(f"   Model parameters: {model_baseline.get_num_params():,}")
    
    # Test 2: SVD on values with reduced rank
    print("\n2. SVD with Reduced Rank")
    model_svd_v = create_model_with_svd(
        use_svd=True, 
        svd_rank=16  # Reduced rank
    ).to(device)
    model_svd_v.eval()
    with torch.no_grad():
        logits_svd_v, _ = model_svd_v(x)
    print(f"   Output shape: {logits_svd_v.shape}")
    print(f"   SVD rank: 16")
    
    # Test 3: SVD with even lower rank
    print("\n3. SVD with Lower Rank")
    model_svd_all = create_model_with_svd(
        use_svd=True,
        svd_rank=8  # Even more reduced rank
    ).to(device)
    model_svd_all.eval()
    with torch.no_grad():
        logits_svd_all, _ = model_svd_all(x)
    print(f"   Output shape: {logits_svd_all.shape}")
    print(f"   SVD rank: 8")
    
    # Test 4: Full rank SVD (no compression)
    print("\n4. Full Rank SVD")
    model_svd_full = create_model_with_svd(
        use_svd=True,
        svd_rank=None  # Full rank
    ).to(device)
    model_svd_full.eval()
    with torch.no_grad():
        logits_svd_full, _ = model_svd_full(x)
    print(f"   Output shape: {logits_svd_full.shape}")
    print(f"   SVD rank: Full")
    
    print("\n" + "=" * 50)
    print("SVD Configuration Test Complete!")

def demonstrate_memory_usage():
    """Demonstrate potential memory savings with SVD"""
    print("\nMemory Usage Comparison:")
    print("=" * 30)
    
    device = 'cpu'  # Use CPU for memory measurement
    
    # Baseline model
    model_baseline = create_model_with_svd(use_svd=False).to(device)
    baseline_params = model_baseline.get_num_params()
    
    # SVD model with rank compression
    model_svd = create_model_with_svd(
        use_svd=True,
        svd_rank=16
    ).to(device)
    svd_params = model_svd.get_num_params()
    
    print(f"Baseline model parameters: {baseline_params:,}")
    print(f"SVD model parameters: {svd_params:,}")
    print(f"Parameter ratio: {svd_params/baseline_params:.2%}")
    print("\nNote: SVD doesn't reduce the actual parameter count,")
    print("but provides computational benefits during attention computation.")

if __name__ == "__main__":
    print("GPT Model with SVD Toggle - Example Usage")
    print("========================================")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test different SVD configurations
    test_svd_configurations()
    
    # Demonstrate memory usage
    demonstrate_memory_usage()
    
    print("\nExample configurations:")
    print("----------------------")
    print("# Disable SVD (baseline)")
    print("config = GPTConfig(use_svd=False)")
    print()
    print("# Enable SVD on value matrices with rank 32")
    print("config = GPTConfig(use_svd=True, svd_rank=32)")
    print()
    print("# Enable SVD with full rank (no compression)")
    print("config = GPTConfig(use_svd=True, svd_rank=None)")
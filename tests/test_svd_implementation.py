#!/usr/bin/env python3
"""
Simple test to verify that SVD toggle actually affects the model computation.
This test compares outputs with and without SVD to ensure they are different.
"""

import torch
from model import GPT, GPTConfig

def test_svd_effect():
    """Test that SVD actually changes the model output"""
    torch.manual_seed(42)
    device = 'cpu'
    
    # Create input
    batch_size = 1
    seq_length = 16
    x = torch.randint(0, 100, (batch_size, seq_length), device=device)
    
    # Small model for quick testing
    base_config = {
        'block_size': 32,
        'vocab_size': 1000,
        'n_layer': 2,
        'n_head': 2,
        'n_embd': 64,
        'dropout': 0.0,
        'bias': False
    }
    
    # Test 1: Model without SVD
    config_no_svd = GPTConfig(**base_config, use_svd=False)
    model_no_svd = GPT(config_no_svd).to(device)
    model_no_svd.eval()
    
    # Test 2: Model with SVD (rank reduction)
    config_svd = GPTConfig(
        **base_config, 
        use_svd=True, 
        svd_rank=8  # Reduced rank should cause different output
    )
    model_svd = GPT(config_svd).to(device)
    
    # Copy weights from no-SVD model to SVD model for fair comparison
    model_svd.load_state_dict(model_no_svd.state_dict())
    model_svd.eval()
    
    # Get outputs
    with torch.no_grad():
        logits_no_svd, _ = model_no_svd(x)
        logits_svd, _ = model_svd(x)
    
    # Calculate difference
    diff = torch.abs(logits_no_svd - logits_svd).max().item()
    
    print("SVD Effect Test")
    print("=" * 20)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits_no_svd.shape}")
    print(f"Max absolute difference: {diff:.6f}")
    
    if diff > 1e-6:
        print("SVD is working - outputs are different")
        return True
    else:
        print("SVD may not be working - outputs are identical")
        return False

def test_svd_toggle():
    """Test that the SVD toggle works correctly"""
    torch.manual_seed(42)
    device = 'cpu'
    
    x = torch.randint(0, 100, (1, 8), device=device)
    
    base_config = {
        'block_size': 16,
        'vocab_size': 1000,
        'n_layer': 1,
        'n_head': 2,
        'n_embd': 32,
        'dropout': 0.0,
        'bias': False
    }
    
    # Test different SVD configurations
    configs = [
        ("No SVD", GPTConfig(**base_config, use_svd=False)),
        ("SVD Rank-4", GPTConfig(**base_config, use_svd=True, svd_rank=4)),
        ("SVD Full Rank", GPTConfig(**base_config, use_svd=True, svd_rank=None)),
    ]
    
    outputs = []
    print("\nSVD Toggle Test")
    print("=" * 20)
    
    for name, config in configs:
        model = GPT(config).to(device)
        if len(outputs) > 0:  # Copy weights for consistent comparison
            model.load_state_dict(outputs[0][1].state_dict())
        model.eval()
        
        with torch.no_grad():
            logits, _ = model(x)
        
        outputs.append((name, model, logits))
        print(f"{name}: output sum = {logits.sum().item():.6f}")
    
    # Compare outputs
    print("\nDifferences:")
    for i in range(1, len(outputs)):
        diff = torch.abs(outputs[0][2] - outputs[i][2]).max().item()
        print(f"{outputs[0][0]} vs {outputs[i][0]}: {diff:.6f}")

if __name__ == "__main__":
    print("SVD Implementation Verification")
    print("=" * 35)
    
    success = test_svd_effect()
    test_svd_toggle()
    
    if success:
        print("\nSVD implementation is working correctly!")
    else:
        print("\nPlease check SVD implementation")
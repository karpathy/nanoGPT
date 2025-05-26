#!/usr/bin/env python3
"""
Simple test script for KronZO implementation.
Tests the basic functionality without full training.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to path to import from lozo_train
sys.path.append('.')

# Import the KronZO functions
from lozo_train import (
    choose_kron_dims, 
    kronzo_perturb_parameters, 
    kronzo_step, 
    kronzo_update,
    kronzo_update_momentum,
    zo_forward
)

# Simple test model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5, bias=False)
        self.linear2 = torch.nn.Linear(5, 1, bias=False)
        
    def forward(self, x, y=None):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        if y is not None:
            loss = torch.nn.functional.mse_loss(x.squeeze(), y)
            return x, loss
        return x

def test_kron_dims():
    """Test Kronecker dimension selection strategies"""
    print("Testing Kronecker dimension selection...")
    
    # Test approx_square strategy
    dims = choose_kron_dims(12, 8, strategy='approx_square')
    print(f"approx_square for (12, 8): {dims}")
    
    # Test fixed_factor strategy
    dims = choose_kron_dims(12, 8, strategy='fixed_factor', max_factor=4)
    print(f"fixed_factor for (12, 8): {dims}")
    
    # Test power2 strategy
    dims = choose_kron_dims(16, 8, strategy='power2')
    print(f"power2 for (16, 8): {dims}")
    
    print("✓ Dimension selection tests passed\n")

def test_kronzo_perturbation():
    """Test KronZO parameter perturbation"""
    print("Testing KronZO parameter perturbation...")
    
    model = SimpleModel()
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Test perturbation
    named_params = kronzo_perturb_parameters(
        model, zo_random_seed=42, step=0, scaling_factor=1, 
        eps=1e-3, strategy='approx_square', max_factor=32
    )
    
    # Check that parameters were actually perturbed
    perturbed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, original_params[name]):
            perturbed = True
            break
    
    assert perturbed, "Parameters should be perturbed"
    print("✓ Parameter perturbation test passed\n")

def test_kronzo_step():
    """Test KronZO gradient estimation step"""
    print("Testing KronZO gradient estimation...")
    
    model = SimpleModel()
    
    # Create simple test data
    x = torch.randn(4, 10)
    y = torch.randn(4)
    
    # Test KronZO step
    loss, projected_grad, named_params = kronzo_step(
        model, x, y, step=0, zo_random_seed=42,
        strategy='approx_square', max_factor=32, eps=1e-3
    )
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert isinstance(projected_grad, float), "Projected grad should be a float"
    assert len(named_params) > 0, "Should return named parameters"
    
    print(f"Loss: {loss.item():.6f}, Projected grad: {projected_grad:.6f}")
    print("✓ KronZO step test passed\n")

def test_kronzo_update():
    """Test KronZO parameter update"""
    print("Testing KronZO parameter update...")
    
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Dummy optimizer
    
    # Store original parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Create named parameters list
    named_params = [(name, param) for name, param in model.named_parameters()]
    
    # Test update
    kronzo_update(
        model, optimizer, projected_grad=-0.1, zo_random_seed=42, 
        step=0, lr=0.01, named_parameters_to_optim=named_params,
        strategy='approx_square', max_factor=32
    )
    
    # Check that parameters were updated
    updated = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, original_params[name]):
            updated = True
            break
    
    assert updated, "Parameters should be updated"
    print("✓ KronZO update test passed\n")

def test_kronzo_momentum():
    """Test KronZO momentum update"""
    print("Testing KronZO momentum update...")
    
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Dummy optimizer
    exp_avg_m = {}  # Empty momentum dict
    
    # Store original parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Create named parameters list
    named_params = [(name, param) for name, param in model.named_parameters()]
    
    # Test momentum update
    kronzo_update_momentum(
        model, optimizer, projected_grad=-0.1, zo_random_seed=42, 
        exp_avg_m=exp_avg_m, step=0, lr=0.01, beta1=0.9,
        named_parameters_to_optim=named_params,
        strategy='approx_square', max_factor=32
    )
    
    # Check that parameters were updated
    updated = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, original_params[name]):
            updated = True
            break
    
    assert updated, "Parameters should be updated"
    assert len(exp_avg_m) > 0, "Momentum dict should be populated"
    print("✓ KronZO momentum test passed\n")

def main():
    """Run all tests"""
    print("Running KronZO tests...\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_kron_dims()
        test_kronzo_perturbation()
        test_kronzo_step()
        test_kronzo_update()
        test_kronzo_momentum()
        
        print("🎉 All KronZO tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
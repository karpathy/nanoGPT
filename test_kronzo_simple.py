#!/usr/bin/env python3
"""
Simple test script for KronZO implementation.
Tests the basic functionality without importing the full training script.
"""

import torch
import numpy as np
import math

# Copy the KronZO functions directly for testing
def find_closest_divisor(n, target):
    """Find divisor of n closest to target"""
    divisors = [i for i in range(1, n+1) if n % i == 0]
    return min(divisors, key=lambda x: abs(x - target))

def choose_kron_dims_approx_square(d_out, d_in):
    """Choose Kronecker factorization dimensions close to square roots"""
    import math
    
    # Find factors closest to square roots
    target_out = int(math.sqrt(d_out))
    target_in = int(math.sqrt(d_in))
    
    # Find divisors closest to targets
    m1 = find_closest_divisor(d_out, target_out)
    n1 = d_out // m1
    
    m2 = find_closest_divisor(d_in, target_in)
    n2 = d_in // m2
    
    return (m1, n1, m2, n2)

def choose_kron_dims_fixed_factor(d_out, d_in, max_factor=32):
    """Choose Kronecker factorization with fixed maximum factor size"""
    # For d_out: choose m1 <= max_factor
    m1 = min(max_factor, d_out)
    # Find largest divisor <= max_factor
    for i in range(m1, 0, -1):
        if d_out % i == 0:
            m1 = i
            break
    n1 = d_out // m1
    
    # For d_in: choose m2 <= max_factor
    m2 = min(max_factor, d_in)
    # Find largest divisor <= max_factor
    for i in range(m2, 0, -1):
        if d_in % i == 0:
            m2 = i
            break
    n2 = d_in // m2
    
    return (m1, n1, m2, n2)

def choose_kron_dims_power2(d_out, d_in):
    def largest_power2_divisor(n):
        """Find the largest power of 2 that divides n"""
        power = 1
        while power * 2 <= n and n % (power * 2) == 0:
            power *= 2
        return power
    
    m1 = largest_power2_divisor(d_out)
    n1 = d_out // m1
    
    m2 = largest_power2_divisor(d_in)
    n2 = d_in // m2
    
    return (m1, n1, m2, n2)

def choose_kron_dims(d_out, d_in, strategy='approx_square', max_factor=32):
    """Choose Kronecker factorization dimensions based on strategy"""
    if strategy == 'approx_square':
        return choose_kron_dims_approx_square(d_out, d_in)
    elif strategy == 'fixed_factor':
        return choose_kron_dims_fixed_factor(d_out, d_in, max_factor)
    elif strategy == 'power2':
        return choose_kron_dims_power2(d_out, d_in)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def kronzo_perturb_parameters(model, zo_random_seed, step, scaling_factor=1, eps=None, strategy='approx_square', max_factor=32):
    """
    Perturb model parameters using Kronecker-structured perturbations.
    
    For each matrix parameter W of shape (d_out, d_in):
    1. Choose factorization dimensions (m1, n1, m2, n2) such that m1*n1 = d_out, m2*n2 = d_in
    2. Sample A ~ N(0, I) of shape (m1, m2) and B ~ N(0, I) of shape (n1, n2)  
    3. Compute Kronecker product: Z = A ⊗ B (shape d_out × d_in)
    4. Perturb: W ← W + scaling_factor * eps * Z
    
    For vector parameters, use standard Gaussian perturbation.
    """
    if eps is None:
        eps = 1e-3  # Default perturbation size
        
    torch.manual_seed(zo_random_seed)
    
    # Create a list of named parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Handle _orig_mod prefix from torch.compile
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use Kronecker-structured perturbation
            d_out, d_in = param.shape[0], param.shape[1]
            
            # Choose Kronecker factorization dimensions
            m1, n1, m2, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
            
            # Sample A and B matrices
            A = torch.randn(m1, m2, device=param.device, dtype=param.dtype)
            B = torch.randn(n1, n2, device=param.device, dtype=param.dtype)
            
            # Compute Kronecker product A ⊗ B
            # kron(A, B) has shape (m1*n1, m2*n2) = (d_out, d_in)
            Z = torch.kron(A, B)
            
            # Apply perturbation
            param.data = param.data + scaling_factor * eps * Z
        else:
            # For vectors (biases), use standard Gaussian perturbation
            z = torch.randn_like(param)
            param.data = param.data + scaling_factor * eps * z
    
    return named_parameters_to_optim

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
    assert len(dims) == 4
    assert dims[0] * dims[1] == 12
    assert dims[2] * dims[3] == 8
    
    # Test fixed_factor strategy
    dims = choose_kron_dims(12, 8, strategy='fixed_factor', max_factor=4)
    print(f"fixed_factor for (12, 8): {dims}")
    assert len(dims) == 4
    assert dims[0] * dims[1] == 12
    assert dims[2] * dims[3] == 8
    assert dims[0] <= 4 and dims[2] <= 4
    
    # Test power2 strategy
    dims = choose_kron_dims(16, 8, strategy='power2')
    print(f"power2 for (16, 8): {dims}")
    assert len(dims) == 4
    assert dims[0] * dims[1] == 16
    assert dims[2] * dims[3] == 8
    
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
    assert len(named_params) > 0, "Should return named parameters"
    print("✓ Parameter perturbation test passed\n")

def test_kronecker_product():
    """Test that Kronecker product has correct dimensions"""
    print("Testing Kronecker product dimensions...")
    
    # Test various matrix sizes
    test_cases = [
        (12, 8),   # 12 = 3*4, 8 = 2*4
        (16, 16),  # 16 = 4*4, 16 = 4*4  
        (6, 10),   # 6 = 2*3, 10 = 2*5
    ]
    
    for d_out, d_in in test_cases:
        m1, n1, m2, n2 = choose_kron_dims(d_out, d_in, strategy='approx_square')
        
        A = torch.randn(m1, m2)
        B = torch.randn(n1, n2)
        Z = torch.kron(A, B)
        
        expected_shape = (d_out, d_in)
        actual_shape = Z.shape
        
        print(f"({d_out}, {d_in}) -> dims({m1}, {n1}, {m2}, {n2}) -> Z.shape {actual_shape}")
        assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
    
    print("✓ Kronecker product dimension tests passed\n")

def main():
    """Run all tests"""
    print("Running KronZO tests...\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_kron_dims()
        test_kronecker_product()
        test_kronzo_perturbation()
        
        print("🎉 All KronZO tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
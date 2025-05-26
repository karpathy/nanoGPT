# KronZO: Kronecker Zero-Order Optimization

KronZO is a novel zero-order optimization method that uses Kronecker-structured perturbations for gradient estimation. This implementation extends the nanoGPT training framework with KronZO support.

## Overview

KronZO addresses the memory limitations of full-rank zero-order methods (like MeZO) while providing more flexibility than low-rank methods (like LoZO). Instead of using low-rank perturbations UV^T, KronZO uses Kronecker-structured perturbations A ⊗ B.

### Key Advantages

1. **Memory Efficiency**: For a matrix of size d_out × d_in, instead of storing d_out × d_in parameters (MeZO) or rank × (d_out + d_in) parameters (LoZO), KronZO stores only (m1 × m2) + (n1 × n2) parameters where m1×n1 = d_out and m2×n2 = d_in.

2. **Full Rank**: Unlike LoZO which is limited by rank, KronZO can represent full-rank perturbations when the Kronecker factors are full rank.

3. **Structured Perturbations**: The Kronecker structure provides a natural way to capture correlations in the parameter space.

## Algorithm

For each matrix parameter W of shape (d_out, d_in):

1. **Factorization**: Choose dimensions (m1, n1, m2, n2) such that m1×n1 = d_out and m2×n2 = d_in
2. **Sampling**: Sample A ~ N(0, I) of shape (m1, m2) and B ~ N(0, I) of shape (n1, n2)
3. **Kronecker Product**: Compute Z = A ⊗ B (shape d_out × d_in)
4. **Perturbation**: W ← W + ε × Z
5. **Gradient Estimation**: Use finite differences: g = (f(θ + εZ) - f(θ - εZ))/(2ε)
6. **Update**: θ ← θ - α × g × Z

## Factorization Strategies

KronZO supports three strategies for choosing Kronecker factorization dimensions:

### 1. Approximate Square (`approx_square`)
- Chooses factors closest to the square roots of the dimensions
- Good balance between memory efficiency and expressiveness
- **Example**: For (12, 8) → (3, 4, 2, 4) since 3×4=12, 2×4=8

### 2. Fixed Factor (`fixed_factor`)
- Limits factor sizes to a maximum value (e.g., 32)
- Provides predictable memory usage
- **Example**: For (12, 8) with max_factor=4 → (4, 3, 4, 2)

### 3. Power of 2 (`power2`)
- Uses the largest power-of-2 divisors
- Efficient for GPU computations
- **Example**: For (16, 8) → (16, 1, 8, 1)

## Usage

### Configuration Files

Two configuration files are provided:

- `config/train_kronzo_shakespeare.py`: For small-scale experiments on Shakespeare dataset
- `config/train_kronzo_owt.py`: For large-scale experiments on OpenWebText dataset

### Key Parameters

```python
# Training method
train_method = 'kronzo'

# KronZO specific parameters
zo_eps = 1e-3                    # Perturbation size
kron_strategy = 'approx_square'  # Factorization strategy
kron_max_factor = 32             # Maximum factor size (for 'fixed_factor' strategy)

# Optional: KronZO with momentum
use_momentum = True              # Enable momentum
momentum_beta = 0.9              # Momentum coefficient

# Optional: Adaptive perturbation size
use_adaptive_eps = True          # Enable adaptive zo_eps
```

### Running KronZO

```bash
# Shakespeare dataset (small model)
python lozo_train.py config=config/train_kronzo_shakespeare.py

# OpenWebText dataset (GPT-2 124M)
python lozo_train.py config=config/train_kronzo_owt.py

# Custom parameters
python lozo_train.py config=config/train_kronzo_shakespeare.py \
    --kron_strategy=fixed_factor --kron_max_factor=16 --use_momentum=True
```

## Implementation Details

### Memory Complexity

For a matrix of size d_out × d_in:

- **MeZO**: O(d_out × d_in) - stores full perturbation
- **LoZO**: O(rank × (d_out + d_in)) - stores low-rank factors
- **KronZO**: O(m1×m2 + n1×n2) where m1×n1 = d_out, m2×n2 = d_in

### Storage Savings Example

For a typical transformer layer with d_out=768, d_in=768:

- **MeZO**: 768 × 768 = 589,824 parameters
- **LoZO** (rank=16): 16 × (768 + 768) = 24,576 parameters  
- **KronZO** (approx_square): 27×28 + 27×28 = 1,512 parameters

KronZO achieves ~390× memory reduction compared to MeZO while maintaining full-rank expressiveness.

### Functions

The main KronZO functions are:

- `kronzo_step()`: Gradient estimation using Kronecker perturbations
- `kronzo_update()`: Parameter update without momentum
- `kronzo_update_momentum()`: Parameter update with momentum
- `kronzo_perturb_parameters()`: Apply Kronecker-structured perturbations
- `choose_kron_dims()`: Select factorization dimensions

## Testing

Run the test suite to verify the implementation:

```bash
python test_kronzo_simple.py
```

This tests:
- Kronecker dimension selection strategies
- Kronecker product computations
- Parameter perturbation functionality

## Comparison with Other Methods

| Method | Memory | Rank | Structure | Expressiveness |
|--------|--------|------|-----------|----------------|
| MeZO | O(d²) | Full | None | Full |
| LoZO | O(rd) | Limited | Low-rank | Limited |
| KronZO | O(√d²) | Full* | Kronecker | High |

*Full rank when Kronecker factors are full rank

## Future Extensions

Potential improvements and extensions:

1. **Adaptive Factorization**: Dynamically adjust factorization during training
2. **Block-Kronecker**: Use block-structured Kronecker products for very large matrices
3. **Hierarchical Kronecker**: Multi-level Kronecker factorizations
4. **Mixed Strategies**: Different strategies for different layer types

## References

This implementation is based on the idea of using Kronecker-structured perturbations for zero-order optimization, extending the concepts from:

- MeZO: Memory-Efficient Zeroth-Order Optimizer
- LoZO: Low-Rank Zeroth-Order Optimizer
- Kronecker-factored approximations in neural networks 
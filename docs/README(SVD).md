# SVD Implementation in GPT Model

This implementation adds Singular Value Decomposition (SVD) capabilities to the GPT model's attention mechanism with a simple toggle system.

## Features

- **Toggle SVD On/Off**: Easy enable/disable of SVD functionality
- **Value Matrix Focus**: Applies SVD only to Value (V) matrices in attention
- **Rank Control**: Configure SVD rank for compression (None for full rank)
- **Standard SVD**: Uses `torch.linalg.svd` for decomposition and reconstruction

## Configuration

### Basic Usage

```python
from model import GPT, GPTConfig

# Disable SVD (baseline)
config = GPTConfig(use_svd=False)

# Enable SVD with default settings (applies to Value matrices)
config = GPTConfig(use_svd=True)

# Enable SVD with rank-32 approximation
config = GPTConfig(use_svd=True, svd_rank=32)
```

### Advanced Configuration

```python
# Enable SVD with different rank settings
config = GPTConfig(use_svd=True, svd_rank=16)    # Rank-16 compression
config = GPTConfig(use_svd=True, svd_rank=None)  # Full rank (no compression)
```

### Training Configuration

Update your training config file (e.g., `config/train_gpt2.py`):

```python
# SVD Configuration
use_svd = True          # Enable SVD on value matrices
svd_rank = 32          # Rank for approximation
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_svd` | bool | False | Master toggle for SVD functionality |
| `svd_rank` | int/None | None | SVD rank (None = full rank) |

## Examples

### Quick Test

```bash
# Run the example script to test different configurations
python example_svd_usage.py

# Train a small model with SVD enabled
python train.py config/train_svd_test.py
```

### Training Comparison

```bash
# Train baseline model (no SVD)
python train.py config/train_gpt2.py

# Edit config/train_gpt2.py to set use_svd = True, then:
python train.py config/train_gpt2.py
```

## How It Works

1. **SVD Decomposition**: Each attention matrix is decomposed as `M = U @ S @ V^T`
2. **Rank Reduction**: Only the top-k singular values are kept for compression
3. **Reconstruction**: The matrix is reconstructed using the reduced components
4. **Value Matrix Focus**: SVD is applied only to Value (V) matrices

### SVD Process

```
Input Matrix (B, nh, T, hs) 
    ↓
Reshape to (B*nh, T, hs)
    ↓
For each batch item:
    SVD: M = U @ S @ V^T
    Truncate to rank k
    Reconstruct: M_k = U_k @ S_k @ V_k^T
    ↓
Reshape back to (B, nh, T, hs)
```

## Performance Considerations

- **Memory**: SVD computation requires additional temporary memory
- **Speed**: SVD adds computational overhead during forward pass
- **Accuracy**: Lower ranks may reduce model accuracy but improve efficiency
- **Batch Size**: SVD is applied per batch item, larger batches = more computation

## Tips for Usage

1. **Start with Full Rank**: Begin with `svd_rank=None` to verify SVD is working
2. **Experiment with Ranks**: Try different ranks (8, 16, 32, 64) to find the sweet spot
3. **Monitor Performance**: Compare loss curves with and without SVD
4. **Small Models First**: Test on smaller models before scaling up

## File Structure

```
├── model.py                 # Main model with SVD implementation
├── config/
│   ├── train_gpt2.py       # Updated with SVD parameters
│   └── train_svd_test.py   # SVD-specific test configuration
├── example_svd_usage.py    # Usage examples and tests
└── SVD_README.md          # This file
```

## Implementation Details

The SVD implementation is contained within the `CausalSelfAttention` class:

- `apply_svd_to_v()`: Applies SVD to Value matrices when enabled
- `_standard_svd_reconstruction()`: Core SVD decomposition and reconstruction using `torch.linalg.svd`

This implementation focuses on simplicity and clarity, applying SVD specifically to Value matrices which are often the most effective target for compression in attention mechanisms.
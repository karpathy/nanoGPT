# Sliding Window Attention

This implementation adds sliding window attention support to nanoGPT, allowing each token to only attend to a limited window of previous tokens rather than the full sequence.

## Overview

Sliding window attention is a modification to the standard self-attention mechanism where each token can only attend to a fixed number of previous tokens (the "window"). This provides several benefits:

1. **Memory Efficiency**: Reduces the quadratic memory complexity for very long sequences
2. **Local Context**: Encourages the model to focus on recent, relevant context
3. **Faster Inference**: Can speed up generation for long sequences
4. **Different Inductive Bias**: Changes how information flows through the network

## Usage

### Configuration

Add the `window_size` parameter to your model configuration:

```python
# In your config file or when creating GPTConfig
window_size = 256  # Each token attends to at most 256 previous tokens
```

- `window_size = None` (default): Full causal attention - each token attends to all previous tokens
- `window_size = N` (integer): Sliding window attention - each token attends to at most N previous tokens

### Example Configuration

See `config/train_shakespeare_sliding_window.py` for a complete example:

```python
from model import GPTConfig, GPT

# Create config with sliding window attention
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=True,
    window_size=256  # Sliding window of 256 tokens
)

model = GPT(config)
```

### Training Example

Train a model with sliding window attention on Shakespeare data:

```bash
python train.py config/train_shakespeare_sliding_window.py
```

You can also override the window size from the command line:

```bash
# Train with window size of 128
python train.py config/train_shakespeare_char.py --window_size=128

# Train with full attention (default)
python train.py config/train_shakespeare_char.py --window_size=None
```

## Implementation Details

### Attention Mask

The sliding window attention mask combines causal masking with a window constraint:

```
For a sequence of length T with window_size W:
- Token at position i can attend to positions [max(0, i-W), i]
- Future positions (> i) are always masked out (causal constraint)
- Positions more than W steps in the past are masked out (window constraint)
```

Example mask for window_size=3 and sequence length 6:

```
     0  1  2  3  4  5  (Key positions)
  0 [1  0  0  0  0  0]
  1 [1  1  0  0  0  0]
Q 2 [1  1  1  0  0  0]
u 3 [0  1  1  1  0  0]  <- Position 3 cannot attend to position 0 (beyond window)
e 4 [0  0  1  1  1  0]
r 5 [0  0  0  1  1  1]
y
```

### Flash Attention Support

The implementation automatically supports both:

1. **Flash Attention** (PyTorch >= 2.0): Uses optimized CUDA kernels with explicit sliding window mask
2. **Manual Implementation**: Pre-computes the sliding window mask as a buffer for efficient reuse

The appropriate implementation is chosen automatically based on PyTorch version.

### Code Changes

The implementation modifies the following in `model.py`:

1. **GPTConfig**: Added `window_size` parameter
2. **CausalSelfAttention.__init__**: Creates sliding window mask if `window_size` is set
3. **CausalSelfAttention.forward**: Applies sliding window mask for both Flash and manual attention

## Performance Considerations

### Memory Usage

- **Full Attention**: O(T²) memory for attention matrix
- **Sliding Window**: O(T × W) effective memory (though full matrix is still allocated)

For very long sequences, consider using efficient implementations that avoid materializing the full attention matrix.

### Computational Complexity

- **Full Attention**: O(T² × D) where D is the embedding dimension
- **Sliding Window**: Same asymptotic complexity, but with reduced effective operations

The main benefit is that gradients don't flow beyond the window, which can speed up training.

### Recommended Window Sizes

- **Small models / short sequences**: Use full attention (`window_size=None`)
- **Character-level models**: 128-512 tokens
- **Word/subword models**: 256-1024 tokens
- **Long context tasks**: 512-2048 tokens

Choose based on your task's typical context dependencies.

## Comparison with Full Attention

### When to Use Sliding Window

- Long sequence generation (documents, code)
- Limited GPU memory
- Tasks where recent context is most important
- Experimentation with different architectural choices

### When to Use Full Attention

- Short sequences (< 512 tokens)
- Tasks requiring long-range dependencies (e.g., question answering)
- Maximum model capacity is desired
- Baseline comparisons

## Example: Comparing Window Sizes

```python
import torch
from model import GPT, GPTConfig

# Full attention model
config_full = GPTConfig(window_size=None)
model_full = GPT(config_full)

# Sliding window model
config_window = GPTConfig(window_size=256)
model_window = GPT(config_window)

# Both models have the same number of parameters
# but different attention patterns
```

## References

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- [Sliding Window Attention in GPT-Neo](https://github.com/EleutherAI/gpt-neo)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

## Future Improvements

Potential enhancements to consider:

1. Efficient sparse attention implementation that avoids full matrix materialization
2. Strided or dilated sliding windows for different layers
3. Combining sliding window with global attention tokens
4. Dynamic window sizes based on sequence length

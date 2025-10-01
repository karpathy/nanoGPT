# Models Package

## Purpose
Neural network architecture and model implementations for ml_playground. Provides modular GPT-based models with proper initialization, inference, and optimization support.

## Structure
- `core/` - Core model implementations (GPT, configuration, inference)
- `layers/` - Reusable neural network layers (attention, MLP, normalization)

## Key APIs
- `GPT` - Main transformer model class
- `GPTConfig` - Model architecture configuration
- `build_gpt_config()` - Configuration factory from experiment config

## Usage Example
```python
from ml_playground.models.core.model import GPT
from ml_playground.models.core.config import GPTConfig

config = GPTConfig(block_size=1024, vocab_size=50000, n_layer=12)
model = GPT(config)
```

## Related Documentation
- [Framework Utilities](../docs/framework_utilities.md) - Model configuration options

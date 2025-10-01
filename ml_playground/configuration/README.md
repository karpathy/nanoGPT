# Configuration Package

## Purpose
Configuration models and loading utilities for ml_playground. Provides strict typing, validation, and TOML-based configuration management with override support.

## Structure
- `models.py` - Pydantic configuration models and validation
- `loading.py` - Configuration loading, merging, and resolution
- `cli.py` - CLI-specific configuration adapters

## Key APIs
- `ExperimentConfig` - Complete experiment configuration
- `TrainerConfig` / `SamplerConfig` / `PreparerConfig` - Section-specific configs
- `load_full_experiment_config()` - Load and merge experiment configuration
- `load_train_config()` / `load_sample_config()` - Partial config loading

## Usage Example
```python
from ml_playground.configuration import load_full_experiment_config

config = load_full_experiment_config(config_path, project_home, experiment_name)
```

## Related Documentation
- [Framework Utilities](../docs/framework_utilities.md) - Configuration structure
- [Development Guidelines](../.dev-guidelines/DEVELOPMENT.md) - Configuration policies

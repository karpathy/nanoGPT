# Configuration Package

## Purpose

Configuration management utilities for `ml_playground`. Provides Pydantic models and loading helpers for experiment
configs.

## Structure

- `models.py` - Pydantic configuration models
- `loading.py` - TOML loading and deep merge helpers

## Key APIs

- `ExperimentConfig` - Complete configuration tree
- `TrainerConfig` - Training-specific configuration
- `load_full_experiment_config()` - Load and validate experiment configuration

## Usage Example

```python
from ml_playground.configuration.loading import load_full_experiment_config

config = load_full_experiment_config(config_path, project_home, experiment_name)
```

## Related Documentation

- [Framework Utilities](../docs/framework_utilities.md) - Configuration guidelines
- [Development Guidelines](../.dev-guidelines/DEVELOPMENT.md) - Configuration policies

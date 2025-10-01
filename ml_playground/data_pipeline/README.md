# Data Pipeline Package

## Purpose
Data pipeline utilities for preparing, transforming, and batching training data in ml_playground experiments. Handles tokenization, data loading, and batch sampling with strict typing and validation.

## Structure
- `preparer.py` - Main data preparation orchestration
- `transforms/` - Data transformation utilities (tokenization, I/O operations)
- `sampling/` - Batch sampling and data loading utilities
- `sources/` - Data source implementations (memory-mapped files, etc.)

## Key APIs
- `create_pipeline()` - Data pipeline factory function
- `PreparationOutcome` - Structured preparation result
- `prepare_with_tokenizer()` - Tokenization and splitting pipeline
- `SimpleBatches` - Training batch iterator
- `MemmapReader` - Memory-efficient data reading

## Usage Example
```python
from ml_playground.data_pipeline import create_pipeline
from ml_playground.configuration.models import PreparerConfig

pipeline = create_pipeline(config, shared_config)
outcome = pipeline.run()
```

## Related Documentation
- [Framework Utilities](../docs/framework_utilities.md) - Data configuration
- [Development Guidelines](../.dev-guidelines/DEVELOPMENT.md) - Data handling standards

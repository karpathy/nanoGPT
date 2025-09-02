# Centralized Framework Utilities

This document describes the centralized framework utilities that have been introduced to provide consistent error handling, progress reporting, and file operations across all experiments in the ml_playground.

## Overview

The ml_playground now includes several centralized utility modules that provide standardized functionality for common operations:

1. `error_handling.py` - Centralized exception classes and error handling utilities
2. `tokenizer.py` - Unified tokenizer protocol and implementations
3. `prepare.py` - Standardized data preparation utilities
4. Updated `config.py`, `trainer.py`, and `sampler.py` with enhanced functionality

## Error Handling Utilities

The `ml_playground/error_handling.py` module provides:

### Exception Classes

- `MLPlaygroundError` - Base exception class for all framework errors
- `ConfigurationError` - Raised for configuration-related errors
- `DataError` - Raised for data-related errors
- `ModelError` - Raised for model-related errors
- `CheckpointError` - Raised for checkpoint-related errors
- `ValidationError` - Raised for validation-related errors
- `FileOperationError` - Raised when file operations fail

### Utility Functions

- `setup_logging()` - Set up a logger with a sensible default configuration
- `safe_call()` - Safely call a function, catching and logging any exceptions
- `safe_file_operation()` - Safely execute a file operation with proper error handling
- `validate_file_exists()` - Validate that a file exists
- `validate_directory_exists()` - Validate that a directory exists
- `validate_config_value()` - Validate a configuration value's type and presence

### Progress Reporting

- `ProgressReporter` - A utility class for reporting progress during long-running operations

## Tokenizer Utilities

The `ml_playground/tokenizer.py` module provides a unified tokenizer protocol and implementations:

### Tokenizer Protocol

All tokenizers implement the `Tokenizer` protocol with these methods:

- `encode(text: str) -> List[int]` - Encode text into a list of token IDs
- `decode(token_ids: List[int]) -> str` - Decode a list of token IDs back into text
- `get_vocab_size() -> int` - Get the vocabulary size
- `get_vocab() -> Dict[str, int]` - Get the vocabulary mapping

### Tokenizer Implementations

- `CharTokenizer` - Character-level tokenizer
- `WordTokenizer` - Word-level tokenizer
- `TiktokenTokenizer` - Tiktoken-based BPE tokenizer

### Factory Function

- `create_tokenizer(tokenizer_type: str, **kwargs) -> Tokenizer` - Factory function to create a tokenizer based on type

## Data Preparation Utilities

The updated `ml_playground/prepare.py` module provides:

### File State Management

- `snapshot_files(paths: Iterable[Path])` - Take a snapshot of file states for diffing later
- `diff_files(paths: Iterable[Path], before: dict[Path, tuple[bool, float, int]])` - Compare file states and determine what changed

### Metadata Creation

- `create_standardized_metadata(tokenizer: Tokenizer, train_tokens: int, val_tokens: int, extras: dict = None)` - Create standardized metadata for dataset preparation

### Data Preparation

- `split_train_val(text: str, split: float = 0.9)` - Split given text into train/val by ratio
- `prepare_with_tokenizer(text: str, tokenizer: Tokenizer, split: float = 0.9)` - Prepare train/val data and metadata using a tokenizer
- `write_bin_and_meta(ds_dir: Path, train: np.ndarray, val: np.ndarray, meta: dict)` - Write train.bin, val.bin, and meta.pkl atomically

## Usage Examples

### Error Handling

```python
from ml_playground.error_handling import DataError, safe_file_operation, validate_file_exists, ProgressReporter

# Validate that a file exists
validate_file_exists(input_file_path, "Input text file")

# Safe file operation
safe_file_operation(lambda: train_arr.tofile(open(train_path, "wb")), logger=logger)

# Progress reporting
progress = ProgressReporter(logger, total_steps=4)
progress.start("Starting dataset preparation")
progress.update(1, "Encoding training data")
progress.finish("Dataset preparation completed")
```

### Tokenizer Usage

```python
from ml_playground.tokenizer import create_tokenizer

# Create a tokenizer
tokenizer = create_tokenizer("tiktoken", encoding_name="gpt2")

# Encode/decode text
train_ids = tokenizer.encode(train_text)
train_text = tokenizer.decode(train_ids)
```

### Data Preparation

```python
from ml_playground.prepare import split_train_val, prepare_with_tokenizer, write_bin_and_meta, snapshot_files, diff_files, create_standardized_metadata

# Split data
train_text, val_text = split_train_val(data)

# Prepare with tokenizer
train_arr, val_arr, meta = prepare_with_tokenizer(data, tokenizer)

# Write files
write_bin_and_meta(ds_dir, train_arr, val_arr, meta)
```

## Benefits

1. **Reduced Code Duplication** - Common functionality is now centralized
2. **Consistent Error Handling** - Standardized exception classes and error messages
3. **Better Progress Reporting** - Unified progress reporting across all experiments
4. **Enhanced Validation** - Comprehensive validation utilities for configuration and data
5. **Improved Maintainability** - Easier to maintain and update since functionality is centralized

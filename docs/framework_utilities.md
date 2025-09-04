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

## CLI Utilities

The `ml_playground/cli.py` module provides the command-line interface for the framework. It uses a standardized structure for defining and running commands.

### Commands

- `prepare` - Prepare a dataset for training
- `train` - Train a model
- `sample` - Sample from a trained model
- `loop` - Run a full train/sample loop

### Usage

```bash
uv run prepare-shakespeare
uv run train-shakespeare-cpu
```

## Configuration System

The framework uses a TOML-based configuration system with dataclasses for validation and type safety. The core configuration is defined in `ml_playground/config.py`.

### Dataclasses

- `ExperimentConfig` - Top-level configuration for an experiment
- `TrainerConfig` - Configuration for the trainer
- `ModelConfig` - Configuration for the model
- `DataConfig` - Configuration for the data

### Overrides

Strict mode: No overrides. Configuration comes solely from TOML files validated by `ml_playground/config.py`.

## Architectural Overview

The framework is designed to be modular and extensible. The core components are:

- **CLI** - Entry point for all operations
- **Configuration** - Manages experiment configuration
- **Data Preparation** - Handles dataset preparation and tokenization
- **Trainer** - Manages the model training loop
- **Sampler** - Handles sampling from a trained model

## Experiment Utilities

The framework provides several utilities for managing experiments:

- `prepare_experiment_dir` - Prepare the output directory for an experiment
- `load_checkpoint` - Load a model checkpoint
- `save_checkpoint` - Save a model checkpoint

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


# Strict Refactoring Guidance (Binding)

The following guidance operationalizes the Developer Guidelines and Import Standards for framework code and tests. Apply these rules immediately and update tests in lockstep.

- **Single Source of Truth for Configuration**
  - Use `ml_playground/config.py` as the only configuration authority.
  - Prefer `load_experiment_toml()` and strongly typed models: `ExperimentConfig`, `TrainerConfig`, `SamplerConfig`, `RuntimeConfig`.
  - Paths must be `pathlib.Path`. Resolve relative paths relative to the TOML file location, not CWD.

- **Public APIs only (no internal/legacy helpers)**
  - Tests and modules must import and use public APIs from concrete submodules, never private helpers in `ml_playground/cli.py` or ad-hoc wrappers.
  - If a test references `cli._internal_*` or `_get_experiment_loader`, delete/replace the test with public-API equivalents.

- **Tokenizer Protocol as the only tokenization entrypoint**
  - Use `ml_playground/tokenizer.py` factory `create_tokenizer()` and the `Tokenizer` protocol.
  - `DataConfig` controls tokenizer selection (`char`, `word`, `tiktoken`) and parameters (e.g., `ngram_size`). Do not re-implement tokenizers in experiments.

- **Centralized Error Handling**
  - Raise and handle exceptions from `ml_playground/error_handling.py`.
  - Use `safe_file_operation()` and `ProgressReporter` for predictable I/O and progress.

- **Import Hygiene (zero workarounds)**
  - Absolute, submodule-level imports only. No umbrella re-exports or star imports. No local imports except documented cycle breaks.
  - Follow `.junie/IMPORT_GUIDELINES.md` strictly.

- **Tests updated first, then code in small steps**
  - For each refactor, update or remove obsolete tests in the same commit. Run all quality gates before committing.


## Deprecations and Removals

The following items are legacy/back-compat and must be removed or migrated:

- Obsolete CLI internal helpers (e.g., `ml_playground.cli._get_experiment_loader`, `cli._resolve_and_load_configs`, `cli._apply_train_overrides`, `cli.load_app_config`). Replace with public functions in `ml_playground/config.py` and explicit CLI flows.
- Ad-hoc config loaders and duplicate path-resolution logic (e.g., alternate `config_loader` modules). Consolidate on `ml_playground/config.py` models and validators.
- Legacy test assumptions about relative paths and implicit defaults. Tests must create minimal TOMLs and assert behavior via typed models.
- Backward-compat CLI flags mutating config. Configuration comes from TOML plus well-defined env JSON overrides only.


## Strict Mode (No Backward Compatibility)

The framework enforces strict behavior across configuration, checkpoints, and tokenizer metadata. Legacy formats and implicit fallbacks are not supported.

- **Configuration (TOML-only, strict schema)**
  - Models in `ml_playground/config.py` are the single source of truth.
  - Unknown/extra keys are forbidden (Pydantic `extra="forbid"`).
  - Only `sample.runtime_ref == "train.runtime"` is supported; no other indirections.
  - No path rewriting or automatic resolution during load; paths are preserved as provided in TOML.

- **Sampler Checkpoints (rotated files only)**
  - Sampler loads checkpoints exclusively via `CheckpointManager` rotated files.
  - Supported patterns:
    - Last: `ckpt_last_XXXXXXXX.pt`
    - Best: `ckpt_best_XXXXXXXX_<metric>.pt`
  - Stable, non-rotated filenames are not used.
  - If no rotated checkpoints exist, a `CheckpointError` is raised.

- **Tokenizer Metadata (required fields)**
  - `meta.pkl` must include `tokenizer_type` in {`char`, `word`, `tiktoken`}.
  - No inference from legacy fields (e.g., `kind`, `stoi`, `itos`, `vocab`). Missing `tokenizer_type` raises `DataError`.
  - For `char`/`word`, vocab fields (`stoi`/`vocab`) are optional but recommended; for `tiktoken`, `encoding_name` defaults to `cl100k_base` when omitted.

These rules ensure deterministic behavior, type safety, and maintainability. Tests and examples have been updated to comply with strict mode.


## Migration Plan (Step-by-Step)

1) Inventory and delete obsolete tests
   - Remove tests referencing private CLI internals or legacy loaders.
   - Keep only tests that exercise public APIs (`config.py`, `trainer.py`, `sampler.py`, `prepare.py`, `tokenizer.py`, `checkpoint.py`).

2) Consolidate configuration
   - Replace any usage of alternate loaders with:
     - `ExperimentConfig = load_experiment_toml(path)`
     - Access `exp.train`, `exp.sample`, and `exp.train.runtime` as typed models.
   - Ensure validators in `config.py` resolve references (e.g., `sample.runtime_ref == "train.runtime"`).

3) Normalize path handling
   - Always resolve relative paths against the TOML file directory.
   - Store resolved paths as `Path` in runtime configs; keep resolution centralized (validators or a small utility), not spread across modules.

4) Tokenization
   - Use `create_tokenizer()` exclusively. Migrate any custom tokenization code into the unified protocol or remove it.
   - Ensure `prepare.py` helpers are used for dataset preparation and metadata creation.

5) Error handling and logging
   - Replace ad-hoc try/excepts with `safe_call()` and `safe_file_operation()` where appropriate.
   - Use `ProgressReporter` for long-running operations in prepare/train/sample loops.

6) CLI simplification
   - Keep CLI as thin wiring: parse args, load TOML via public API, invoke `prepare`, `trainer`, `sampler` modules.
   - Remove internal shims and re-export patterns. Public surface is the concrete module functions/classes.

7) Quality gates per step
   - After each small refactor, run:
     - `uv run ruff check --fix . && uv run ruff format .`
     - `uv run pyright && uv run mypy ml_playground`
     - `uv run pytest -n auto -W error --strict-markers --strict-config -v`


## Test Refactoring Checklist

- Replace imports of private CLI helpers with public APIs.
- Construct minimal TOML files in tests; avoid implicit global defaults unless explicitly passed.
- Validate types and error messages from Pydantic models directly (no monkeypatching internals).
- Use `tmp_path` for any filesystem needs; assert on files created by `prepare.write_bin_and_meta()` and checkpoint routines.
- Prefer parametrized tests for schedules, learning-rate helpers, and tokenizers.


## Example: Loading and Resolving an Experiment

```python
from pathlib import Path
from ml_playground.config import load_experiment_toml, SamplerConfig

p = Path("experiments/exp/config.toml")
exp = load_experiment_toml(p)

# sample.runtime may be provided directly or via reference to train.runtime
sample_cfg: SamplerConfig = exp.sample
runtime = sample_cfg.runtime  # already resolved by model validator when runtime_ref == "train.runtime"

assert runtime.out_dir.is_absolute()
assert runtime.device in {"cpu", "mps", "cuda"}
```


## Example: Tokenizer and Preparation Flow

```python
from pathlib import Path
from ml_playground.tokenizer import create_tokenizer
from ml_playground.prepare import prepare_with_tokenizer, write_bin_and_meta

tok = create_tokenizer("tiktoken", encoding_name="gpt2")
train_arr, val_arr, meta = prepare_with_tokenizer("hello world", tok)
ds_dir = Path("/tmp/dataset")
write_bin_and_meta(ds_dir, train_arr, val_arr, meta)
```


## Enforcement

- These rules extend `.junie/DEVELOPMENT.md` and `.junie/IMPORT_GUIDELINES.md` and are binding.
- Refactors must remove legacy/back-compat code and migrate tests simultaneously.
- All quality gates must pass locally before PR.

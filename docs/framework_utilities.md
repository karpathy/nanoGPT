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

### Data Preparation Example

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

Run subcommands via Make targets (preferred):

```bash
# Prepare data for an experiment
make prepare <experiment> [CONFIG=/path/to/config.toml]

# Train a model for an experiment
make train <experiment> CONFIG=/path/to/config.toml

# Sample from a trained model
make sample <experiment> CONFIG=/path/to/config.toml

# Run prepare -> train -> sample in one go
make loop <experiment> CONFIG=/path/to/config.toml
```

Output is quieter by default due to a global `.SILENT:` in the Makefile; only explicit messages are printed.

## Configuration System

The framework uses a TOML-based configuration system with strict Pydantic models for validation and type safety. The core configuration is defined in `ml_playground/config.py`.

### Config Models

- `ExperimentConfig` — Top-level configuration for an experiment (`prepare`, `train`, `sample`).
- `TrainerConfig` — Groups `ModelConfig`, `DataConfig`, `OptimConfig`, `LRSchedule`, and `RuntimeConfig`.
- `SamplerConfig` — Requires a `RuntimeConfig` and a `SampleConfig`.
- `ModelConfig` — Architecture parameters (layers, heads, embedding, block size, dropout, etc.).
- `DataConfig` — Dataset directory, batch/block sizes, tokenizer selection, and dataset knobs.
- `OptimConfig` — Optimizer hyperparameters.
- `LRSchedule` — Learning-rate schedule knobs (warmup/decay).
- `RuntimeConfig` — Output paths, device/dtype, seeding, checkpoint policy, logging, and loop cadence.
- `SampleConfig` — Inference-time sampling knobs.

All config models inherit from `_FrozenStrictModel` (immutable, `extra="forbid"`). `ExperimentConfig` always carries a `logger` (injected via default factory or overridden by the loader/CLI); section models do not define their own logger.

### Validated Type Aliases

Common numeric constraints are expressed via annotated aliases:

| Alias                         | Constraint                   | Example uses                                 |
|-------------------------------|------------------------------|----------------------------------------------|
| `AtLeastOneInt`               | `int >= 1`                   | `RuntimeConfig.eval_interval`, `DataConfig.batch_size` |
| `NonNegativeStrictInt`        | `int >= 0`                   | `RuntimeConfig.max_iters`, `ckpt_top_k`      |
| `PositiveStrictInt`           | `int > 0`                    | `ModelConfig.n_layer`, `n_head`, `n_embd`    |
| `SeedInt`                     | `int >= 0`                   | `RuntimeConfig.seed`                          |
| `MinutesNonNegative`          | `int >= 0`                   | `RuntimeConfig.ckpt_time_interval_minutes`    |
| `UnitIntervalStrictFloat`     | `0.0 <= float <= 1.0`        | `RuntimeConfig.best_smoothing_alpha`, `ema_decay` |
| `PosUnitIntervalStrictFloat`  | `0.0 < float <= 1.0`         | `SampleConfig.top_p`                          |
| `NonNaNNonNegativeStrictFloat`| `float >= 0.0` and not NaN   | `OptimConfig.grad_clip`, `weight_decay`       |
| `EpochCount`                  | alias of `AtLeastOneInt`     | `RuntimeConfig.iters_per_epoch`, `max_epochs` |

Examples:

- `RuntimeConfig.eval_interval: AtLeastOneInt`
- `DataConfig.batch_size: AtLeastOneInt`
- `ModelConfig.n_layer: PositiveStrictInt`

### Path Handling

- Path fields (e.g., `dataset_dir`, `raw_dir`, `out_dir`) are `pathlib.Path` in models.
- `SharedConfig` is the single authority for resolving project-scoped paths. A `@model_validator(before=True)` resolves `project_home`, `dataset_dir`, `train_out_dir`, and `sample_out_dir` relative to `config_path` when provided as strings or relative Paths.
- Section models (`PreparerConfig`, `TrainerConfig`, `SamplerConfig`) no longer carry ad-hoc path resolution; they accept already-normalized values. Minimal relative resolution remains for section-local fields when explicitly needed and context is available.

### Logger Behavior

- `ExperimentConfig` always carries a logger via a default factory; the loader/CLI may override it.
- Section configs (`PreparerConfig`, `TrainerConfig`, `SamplerConfig`) do not define their own logger fields; they inherit common behavior from `_FrozenStrictModel` but remain free of logger extras in the schema.

### Strictness and Unknown Keys

- Unknown keys raise validation errors (`extra="forbid"`).
- String coercions for paths are not accepted at the model layer; only loaders may coerce TOML strings to `Path`.
- `SamplerConfig.runtime` is required — no reference indirections are supported.

### Cross-Field Validations

Config validators enforce a few important invariants:

- `train.data.block_size <= train.model.block_size`.
- If `train.schedule.decay_lr == true` then `train.schedule.min_lr <= train.optim.learning_rate`.
- If `train.schedule.decay_lr == false` then `train.schedule.warmup_iters == 0`.
- For tokenization: when `train.data.tokenizer == "tiktoken"`, enforce `train.data.ngram_size == 1`.
- `RuntimeConfig` cadence and timing fields are range-checked (e.g., `eval_interval >= 1`, `ckpt_time_interval_minutes >= 0`, etc.).

These validations fail fast with descriptive errors to catch misconfigurations early.

## Common Pitfalls

- __Relative paths in TOML are not auto-resolved by models__
  - Models accept `Path` only; loaders resolve TOML strings relative to the config file directory. If constructing models directly in code, pass `Path` objects.

- __Missing `[sample.runtime]` section__
  - `SamplerConfig.runtime` is required; no `runtime_ref` indirections are supported. Define `[sample.runtime]` explicitly in TOML.

- __Using `tiktoken` with `ngram_size > 1`__
  - For `DataConfig`, when `tokenizer == "tiktoken"`, set `ngram_size = 1`.

- __Scheduler warmup with `decay_lr = false`__
  - If LR decay is disabled, set `warmup_iters = 0`.

- __`min_lr` higher than `learning_rate` when decay is enabled__
  - Ensure `schedule.min_lr <= optim.learning_rate` when `schedule.decay_lr = true`.

- __`max_iters = 0` for eval-only/smoke flows__
  - Allowed. `RuntimeConfig.max_iters` can be 0; ensure `eval_interval`, `eval_iters`, and `log_interval` are >= 1.

- __Checkpoint rotation knobs inert by default__
  - `ckpt_top_k = 0` disables top-k pruning. Set `ckpt_top_k > 0` and ensure metric settings are correct (`ckpt_metric`, `ckpt_greater_is_better`).

- __Unknown keys in TOML__
  - All models use `extra="forbid"`. Remove unknown keys or add them under `extras` where appropriate.

- __Logger availability__
  - `ExperimentConfig` always has a logger (default factory or loader/CLI override). Section configs do not accept a logger field; avoid injecting extras into sections.

## Architectural Overview

The framework is designed to be modular and extensible. The core components are:

- __CLI__ - Entry point for all operations
- __Configuration__ - Manages experiment configuration
- __Data Preparation__ - Handles dataset preparation and tokenization
- __Trainer__ - Manages the model training loop
- __Sampler__ - Handles sampling from a trained model

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

### Model Construction

```python
from ml_playground.models.core.config import GPTConfig
from ml_playground.models.core.model import GPT

gpt_cfg = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
)
model = GPT(gpt_cfg, logger=exp.logger)
```

## Benefits

1. __Reduced Code Duplication__ - Common functionality is now centralized
2. __Consistent Error Handling__ - Standardized exception classes and error messages
3. __Better Progress Reporting__ - Unified progress reporting across all experiments
4. __Enhanced Validation__ - Comprehensive validation utilities for configuration and data
5. __Improved Maintainability__ - Easier to maintain and update since functionality is centralized

## Strict Refactoring Guidance (Binding)

The following guidance operationalizes the Developer Guidelines and Import Standards for framework code and tests. Apply these rules immediately and update tests in lockstep.

- __Single Source of Truth for Configuration__
  - Use `ml_playground/config.py` as the only configuration authority.
  - Prefer `load_experiment_toml()` and strongly typed models: `ExperimentConfig`, `TrainerConfig`, `SamplerConfig`, `RuntimeConfig`.
  - Paths must be `pathlib.Path`. Resolve relative paths relative to the TOML file location, not CWD.

- __Public APIs only (no internal/legacy helpers)__
  - Tests and modules must import and use public APIs from concrete submodules, never private helpers in `ml_playground/cli.py` or ad-hoc wrappers.
  - If a test references `cli._internal_*` or `_get_experiment_loader`, delete/replace the test with public-API equivalents.

- __Tokenizer Protocol as the only tokenization entrypoint__
  - Use `ml_playground/tokenizer.py` factory `create_tokenizer()` and the `Tokenizer` protocol.
  - `DataConfig` controls tokenizer selection (`char`, `word`, `tiktoken`) and parameters (e.g., `ngram_size`). Do not re-implement tokenizers in experiments.

- __Centralized Error Handling__
  - Raise and handle exceptions from `ml_playground/error_handling.py`.
  - Use `safe_file_operation()` and `ProgressReporter` for predictable I/O and progress.

- __Import Hygiene (zero workarounds)__
  - Absolute, submodule-level imports only. No umbrella re-exports or star imports. No local imports except documented cycle breaks.
  - Follow `.dev-guidelines/Readme.md` (Import Standards) strictly.

- __Tests updated first, then code in small steps__
  - For each refactor, update or remove obsolete tests in the same commit. Run all quality gates before committing.

## Deprecations and Removals

The following items are legacy/back-compat and must be removed or migrated:

- Obsolete CLI internal helpers (e.g., `ml_playground.cli._get_experiment_loader`, `cli._resolve_and_load_configs`, `cli._apply_train_overrides`, `cli.load_app_config`). Replace with public functions in `ml_playground/config.py` and explicit CLI flows.
- Ad-hoc config loaders and duplicate path-resolution logic (e.g., alternate `config_loader` modules). Consolidate on `ml_playground/config.py` models and validators.
- Legacy test assumptions about relative paths and implicit defaults. Tests must create minimal TOMLs and assert behavior via typed models.
- Backward-compat CLI flags mutating config. Configuration comes from TOML plus well-defined env JSON overrides only.

## Strict Mode (No Backward Compatibility)

The framework enforces strict behavior across configuration, checkpoints, and tokenizer metadata. Legacy formats and implicit fallbacks are not supported.

- __Configuration (TOML-only, strict schema)__
  - Models in `ml_playground/config.py` are the single source of truth.
  - Unknown/extra keys are forbidden (Pydantic `extra="forbid"`).
  - No runtime references/indirections are supported; `SamplerConfig.runtime` must be provided explicitly in TOML.
  - Path resolution and coercion to `Path` happen in loaders relative to the TOML file; models themselves only accept `Path`.

- __Sampler Checkpoints (rotated files only)__
  - Sampler loads checkpoints exclusively via `CheckpointManager` rotated files.
  - Supported patterns:
    - Last: `ckpt_last_XXXXXXXX.pt`
    - Best: `ckpt_best_XXXXXXXX_<metric>.pt`
  - Stable, non-rotated filenames are not used.
  - If no rotated checkpoints exist, a `CheckpointError` is raised.

- __Tokenizer Metadata (required fields)__
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
   - Ensure validation uses strict models; no reference-resolution mechanics are supported.

3) Normalize path handling
   - Resolve relative paths against the TOML file directory via `SharedConfig`'s pre-validator. Downstream modules must treat these paths as canonical and avoid re-resolving.

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

## Example: Loading an Experiment

```python
from pathlib import Path
from ml_playground.config import load_experiment_toml, SamplerConfig

p = Path("experiments/exp/config.toml")
exp = load_experiment_toml(p)

sample_cfg: SamplerConfig = exp.sample
runtime = sample_cfg.runtime  # explicit in TOML; no reference resolution

assert runtime.out_dir.is_absolute() or not runtime.out_dir.is_absolute()
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

- These rules extend `.dev-guidelines/DEVELOPMENT.md` and `.dev-guidelines/IMPORT_GUIDELINES.md` and are binding.
- Refactors must remove legacy/back-compat code and migrate tests simultaneously.
- All quality gates must pass locally before PR.

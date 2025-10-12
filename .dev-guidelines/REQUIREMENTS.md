---
trigger: model_decision
description: when talking about requirements
---

# Checkpointing System Requirements

## Overview

The checkpointing system manages model snapshots during training to enable resuming training and model evaluation. The
system should be strict and well-defined with clear behavior.

## Configuration

### Checkpointing Policy

- Checkpoints are managed through `RuntimeConfig.checkpointing`
- Two separate policies for last and best checkpoints:
  - `checkpointing.keep.last`: Number of last checkpoints to keep (default: 1)
  - `checkpointing.keep.best`: Number of best checkpoints to keep (default: 1)
- Both values must be >= 0

### Checkpoint Files

- Rotated-only checkpointing is enforced.
- Supported patterns:
  - Last: `ckpt_last_XXXXXXXX.pt`
  - Best: `ckpt_best_XXXXXXXX_<metric>.pt`

## Behavior

### Initialization

- First rotated checkpoint is always saved immediately after model initialization

### During Training

- Last rotated checkpoints are saved periodically based on training progress
- Best rotated checkpoints are saved whenever the validation metric improves

### Checkpoint Management

- Strict enforcement of keep policies:
  - Keep exactly `keep.last` last checkpoints (0 means none)
  - Keep exactly `keep.best` best checkpoints (0 means none)
- No fallback behavior - strict validation of configuration
- No legacy options supported

### Checkpoint Structure

- Checkpoints should be strongly typed objects, not dictionaries
- Each checkpoint contains:
  - Model state dictionary
  - Optimizer state dictionary
  - Model arguments
  - Current iteration number
  - Best validation loss
  - Configuration
  - Optional EMA (Exponential Moving Average) shadow weights

## Error Handling

- Strict failure modes - no silent failures
- Clear error messages for misconfiguration
- Validation of checkpoint files on load

## Filesystem Access and Path Suppliers

- The configuration loader (`ml_playground.config_loader`) is the single boundary that interacts with the filesystem for
  configuration concerns (parsing TOML, resolving and coercing paths, injecting logger).
  - Reads TOML files (defaults and experiment config) using strict UTF-8 parsing.
  - Resolves known relative paths relative to the config file directory when appropriate.
  - Coerces string paths to `pathlib.Path` objects before validation.
  - Injects required runtime context (e.g., logger) prior to strict Pydantic validation.
- All other modules must treat `Path` values as already validated inputs and must not perform additional path resolution
  or ad-hoc filesystem probing beyond their explicit responsibilities (e.g., training writing checkpoints, sampler
  reading checkpoints). Runtime commands may perform existence checks for required runtime artifacts (see Universal Meta
  Requirement).
- No fallback or legacy behavior: invalid or missing paths must surface as early errors during load/validation.
- To keep call sites simple and decoupled from path layout, the loader should expose supplier functions (thin helpers)
  that return the canonical `Path` objects needed by downstream components. Examples:
  - `get_cfg_path(experiment: str, exp_config: Path | None) -> Path` (already provided)
  - `get_default_config_path(config_path: Path) -> Path` (already provided)
  - If needed, additional suppliers for derived locations (e.g., experiment directories) should live in `config_loader`
    and not be re-implemented elsewhere.

## Universal Meta Requirement

- `meta.pkl` is a required data artifact for training and sampling.
  - `DataConfig.meta_pkl` is mandatory (default: `"meta.pkl"`).
  - `DataConfig.meta_path` always resolves to a `Path` under `dataset_dir`.
- Responsibility split:
  - Prepare: must write `meta.pkl` to the dataset directory.
  - Loader: remains pure wrt data artifacts; it does not check for the presence of `meta.pkl`.
  - CLI (runtime): enforces fast-fail validations for required runtime artifacts:
    - Train: requires `train.data.meta_path` to exist.
    - Sample: requires `train.data.meta_path` or `<sample.runtime.out_dir>/<experiment>/meta.pkl` to exist.
    - Loop: skipping prepare requires `train.bin`, `val.bin`, and `meta.pkl` to be present.
  - Rationale: keeps config loading deterministic and testable without a filesystem, while still failing early at
    execution time when artifacts are missing.

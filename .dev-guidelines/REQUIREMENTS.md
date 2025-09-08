---
trigger: model_decision
description: when talking about requirements (checkpointing behavior)
globs: *.py, *.md
---

# Checkpointing System Requirements

## Overview

The checkpointing system manages model snapshots during training to enable resuming training and model evaluation. The system should be strict and well-defined with clear behavior.

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

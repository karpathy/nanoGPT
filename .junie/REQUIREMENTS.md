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
- `ckpt_last.pt`: Last checkpoint (periodically updated during training)
- `ckpt_best.pt`: Best checkpoint (updated when validation metric improves)

## Behavior

### Initialization
- First checkpoint is always saved immediately after model initialization
- Both `ckpt_last.pt` and `ckpt_best.pt` are created with the same initial model state

### During Training
- Last checkpoint (`ckpt_last.pt`) is updated periodically based on training progress
- Best checkpoint (`ckpt_best.pt`) is updated whenever the validation metric improves
- When saving a checkpoint:
  - If it's the best so far, update `ckpt_best.pt`
  - Always update `ckpt_last.pt` (unless it's the same iteration as the best)

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

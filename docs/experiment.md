# Experiment Protocol

This document summarizes the current experiment process for this repository.
Read this together with `docs/research.md` before running any experiment.

## Goal

The research goal is to test whether a line-search-based learning rate scheduler can
adaptively tune learning rate across different model sizes, optimizers, and model
architectures. The main comparison is the wall-clock time required to go from
hyperparameter search to reaching a preset target metric.

## Core Rule

Only tune the learning rate. Do not change any other hyperparameters unless there is
an explicit instruction to do so.

## Hyperparameter Optimization

- Framework: Optuna
- Sampler: TPE (default Optuna sampler)
- Search controller: serial successive halving
- Search space: learning rate only

Example config:

```yaml
hyperparameters:
  learning_rate:
    type: log_uniform
    range: [1e-5, 1e-3]
```

All other hyperparameters must stay fixed for a given experiment.

## Task Configuration

Task settings should be defined in the experiment config file under `config/experiments/`.

Expected fields:

```yaml
task:
  train_metric: ...
  test_metric: ...
  target:
    type: threshold / improvement / custom
    value: ...
  max_running_time_per_trial: ...
```

In the current implementation, the target is represented as explicit train and test
thresholds plus a metric mode:

```yaml
task:
  train_metric: train_loss
  test_metric: val_loss
  metric_mode: min
  train_target: ...
  test_target: ...
  max_running_time_per_trial_hours: ...
```

## Trial Execution

Optuna search:
Use Optuna to search only the learning rate within the predefined learning rate range.
Each trial is limited by `task.max_running_time_per_trial_hours`.
All other hyperparameters must remain fixed.

Serial successive halving:
The current staged search is not a single Optuna study with in-process pruning.
Instead, `run_stage1_optuna.py` acts as a serial controller over explicit halving levels.

Level construction:
- The tuning budgets come from `optuna.max_study_time_hours_levels`.
- The controller derives one iteration budget per level from
  `task.num_iterations_per_trial`.
- With reduction factor 4, the first level starts with `4^(L-1)` trials for `L` levels.

Per-level behavior:
1. Launch every active trial for the current rung budget.
2. Persist trial artifacts under `shared_trials/<trial_id>/`.
3. Snapshot machine-readable outputs into `level_x/stage1/trial_snapshots/<trial_id>/`.
4. Rank trials by `task.train_metric`.
5. Prune the bottom 3/4 of trials.
6. Resume the surviving 1/4 from checkpoint for the next level.

Resume semantics:
- The first rung runs each trial with `init_from=scratch`.
- Later rungs run survivors with `init_from=resume`.
- Resume happens from the checkpoint already stored in that trial's shared directory.
- The resume path must accept `ckpt.pt` when present and fall back to `ckpt_last.pt`
  when only the last checkpoint exists.

Final-result selection:
There is no separate second training pass anymore.
Instead, after each Optuna run finishes, the last completed non-pruned trial is promoted
to the final result for that tuning budget.

Output compatibility:
The repository still writes `stage1_result.json`, `stage2_result.json`,
`stage2/final/summary.json`, and related manifest files so downstream scripts can keep
using the same paths as before.

Early stopping behavior:
Pruning happens only at rung boundaries, after all active trials for that level have finished.
There is no mid-rung pruning. Within a rung, evaluation still happens at the configured
evaluation interval and any checkpoint written during the rung is part of the trial state
that may later be resumed.

## Logging

Track the following at each logging or evaluation interval:

- `step`
- `train_loss`
- `train_metric`
- `test_loss`
- `test_metric`
- `learning_rate`
- `wall_clock_time`

In the current implementation, Optuna pruning decisions are made from the reported
evaluation metric at each eval step.

## Checkpointing

Checkpointing is required for serial successive halving.

Stage-one trial layout:
- Each trial keeps a persistent working directory at `shared_trials/<trial_id>/`.
- This directory is reused across levels for the same trial.
- Checkpoints, `summary.json`, `records.jsonl`, and `trial.log` live in that directory
  while the trial is active.

Checkpoint policy:
- The staged search should save at least a resumable last checkpoint for every trial.
- A best checkpoint may also exist, but the resume path must not depend on it.
- After a level finishes, survivors continue from the saved checkpoint in their existing
  shared trial directory.

Level snapshots and promoted outputs:
- After each rung execution, the controller copies summary/log/records into
  `level_x/stage1/trial_snapshots/`.
- The selected non-pruned trial is promoted into the stage-two-compatible layout:
  `stage1_result.json`, `stage2_result.json`, `stage2/final/summary.json`, and manifests.

## Time Definition

Time is defined as forward wall-clock time plus backward wall-clock time.

In practice, report elapsed wall-clock time for the full training run and the time at
which the configured target metric is first reached.

## Summary Table Labels

When generating the experiment summary table:

- Serial-halving baselines keep their optimizer names such as `cosine`, `muon`, and
  `schedulefree_adam`.
- Direct line-search runs must be split into two separate rows:
  `linesearch_adam` and `linesearch_muon`.
- These direct line-search rows do not have a tuning phase, so the tuning-time cell is
  rendered as `Nah`.





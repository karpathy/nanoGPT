---
trigger: always_on
description: Quick start instructions for preparing the ml_playground development environment
---

# ml_playground Setup Guide

Quick start guide for setting up the ml_playground development environment.

<details>
<summary>Related documentation</summary>

- [Developer Guidelines Index](./Readme.md) – Entry point for principles, quick-start commands, and reference links.
- [Development Practices](./DEVELOPMENT.md) – Quality gates, tooling workflows, and architecture notes.
- [Documentation Guidelines](./DOCUMENTATION.md) – Shared Markdown structure and cross-referencing standards.

</details>

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Basic Workflow Commands](#basic-workflow-commands)
- [Configuration System](#configuration-system)
- [Testing](#testing)
- [Quality Gates](#quality-gates)
- [Quick Troubleshooting](#quick-troubleshooting)

## Prerequisites

- Python version: see pyproject.toml (currently ">=3.13")
- UV package manager installed

## Environment Setup

### 1. Create Virtual Environment

```bash
uv venv
uv sync --all-groups
```

### 2. Verify Installation

```bash
uv run python -c "import ml_playground; print('✓ ml_playground import OK')"
```

## Basic Workflow Commands

### Dataset Preparation

```bash
uv run cli --exp-config src/ml_playground/experiments/shakespeare/config.toml prepare shakespeare
uv run cli --exp-config src/ml_playground/experiments/bundestag_char/config.toml prepare bundestag_char
```

Notes:

- The prepare step is responsible for writing `meta.pkl` into your dataset directory alongside `train.bin` and
  `val.bin`.
- If you use a custom preparer, ensure it calls `ml_playground.prepare.write_bin_and_meta(...)` or equivalent
  utilities which always write a standardized `meta.pkl`.

### Training

```bash
uv run cli --exp-config src/ml_playground/experiments/shakespeare/config.toml train shakespeare
uv run cli --exp-config src/ml_playground/experiments/bundestag_char/config.toml train bundestag_char
```

Notes:

- Universal meta policy: training requires `meta.pkl` to exist at `train.data.meta_path` (usually
  `<dataset_dir>/meta.pkl`).
- The CLI will fail fast with a clear error if `meta.pkl` is missing.

### Sampling

```bash
uv run cli --exp-config src/ml_playground/experiments/shakespeare/config.toml sample shakespeare
uv run cli --exp-config src/ml_playground/experiments/bundestag_char/config.toml sample bundestag_char
```

Notes:

- Sampling requires `meta.pkl` to exist either at `train.data.meta_path` or under the sample runtime output directory at
  `<out_dir>/<experiment>/meta.pkl`.
- The CLI will fail fast with a clear error if `meta.pkl` cannot be found in one of the expected locations.

### End-to-End Workflow

- Run `prepare`, `train`, and `sample` sequentially with the commands above.

## Configuration System

- All configuration via TOML — __Single Source of Truth for Configuration__
  - Use the `ml_playground/configuration/` package as the only configuration authority (`models`, `loading`, `cli`).
  - Prefer `configuration.loading.load_experiment_toml()` (or `load_full_experiment_config`) and strongly typed models:
    `ExperimentConfig`, `TrainerConfig`, `SamplerConfig`, `RuntimeConfig`.
  - Global CLI option `--exp-config PATH` selects an alternative experiment TOML file (replaces the experiment’s
    `config.toml`). The global `experiments/default_config.toml` is still merged first under the experiment config.
  - Environment JSON overrides are supported, deep-merged, and strictly re-validated; invalid overrides are ignored to
    avoid breaking flows:
    - `ML_PLAYGROUND_TRAIN_OVERRIDES`
    - `ML_PLAYGROUND_SAMPLE_OVERRIDES`
- Paths automatically converted to `pathlib.Path` objects

## Testing

- Unit tests: see `tests/unit/README.md` (fast, isolated; no external TOML)
- Integration tests: see `tests/integration/README.md` (compose small real components via Python APIs)
- End-to-end (E2E) tests: see `tests/e2e/README.md` (CLI wiring, config merge, logging)
  - When invoking the CLI in E2E tests, pass the tiny test defaults explicitly:
    `--exp-config tests/e2e/ml_playground/experiments/test_default_config.toml`
- Follow strict TDD as outlined in [`TESTING.md`](./TESTING.md#test-driven-development-required) and commit pairing rules from [`DEVELOPMENT.md`](./DEVELOPMENT.md#commit-standards).

## Quality Gates

```bash
# Full gate: ruff (lint+format), pyright, mypy, pytest
uv run ci-tasks quality

# Extended: optional mutation testing (Cosmic Ray)
uv run ci-tasks quality-ext
```

## Quick Troubleshooting

__Tests cannot import `ml_playground`__: You're not in the project venv - run `uv venv` then `uv sync --all-groups`

__Missing pytest__: Run `uv sync --all-groups` to install dev dependencies

__UV hangs during venv creation__: Exit any active virtual environment first, then retry

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

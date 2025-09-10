---
trigger: always_on
description: 
globs: 
---

# ml_playground Setup Guide

Quick start guide for setting up the ml_playground development environment.

## Prerequisites

- Python version: see pyproject.toml (currently "<3.13")
- UV package manager installed

## Environment Setup

### 1. Create Virtual Environment
```bash
make setup
```

### 2. Verify Installation
```bash
make verify
```

## Basic Workflow Commands

### Dataset Preparation
```bash
make prepare EXP=shakespeare
make prepare EXP=bundestag_char
```

### Training
```bash
make train EXP=shakespeare CONFIG=ml_playground/configs/shakespeare_cpu.toml
make train EXP=bundestag_char CONFIG=ml_playground/configs/bundestag_char_cpu.toml
```

### Sampling
```bash
make sample EXP=shakespeare CONFIG=ml_playground/configs/shakespeare_cpu.toml
make sample EXP=bundestag_char CONFIG=ml_playground/configs/bundestag_char_cpu.toml
```

### End-to-End Loop
```bash
make loop EXP=bundestag_char CONFIG=ml_playground/configs/bundestag_char_cpu.toml
```

## Configuration System

- All configuration via TOML files mapped to dataclasses in `ml_playground/config.py`
- No ad-hoc CLI parameter overrides — TOML remains the source of truth.
- Allowed exceptions (as implemented):
  - Global CLI option `--exp-config PATH` selects an alternative experiment TOML file (replaces the experiment’s `config.toml`). The global `experiments/default_config.toml` is still merged first under the experiment config.
  - Environment JSON overrides are supported, deep-merged, and strictly re-validated; invalid overrides are ignored to avoid breaking flows:
    - `ML_PLAYGROUND_TRAIN_OVERRIDES`
    - `ML_PLAYGROUND_SAMPLE_OVERRIDES`
- Paths automatically converted to `pathlib.Path` objects
- Device defaults to CPU-first; MPS/CUDA supported when explicitly configured

## Testing

- Unit tests: see `tests/unit/README.md` (fast, isolated; no external TOML)
- Integration tests: see `tests/integration/README.md` (compose small real components via Python APIs)
- End-to-end (E2E) tests: see `tests/e2e/README.md` (CLI wiring, config merge, logging)
  - When invoking the CLI in E2E tests, pass the tiny test defaults explicitly:
    `--exp-config tests/e2e/ml_playground/experiments/test_default_config.toml`

## Quick Troubleshooting

**Tests cannot import `ml_playground`**: You're not in the project venv - run `uv venv` then `uv sync --all-groups`

**Missing pytest**: Run `uv sync --all-groups` to install dev dependencies

**UV hangs during venv creation**: Exit any active virtual environment first, then retry

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
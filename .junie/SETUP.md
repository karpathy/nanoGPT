# ml_playground Setup Guide

Quick start guide for setting up the ml_playground development environment.

## Prerequisites

- Python version: see pyproject.toml (currently "<3.13")
- UV package manager installed

## Environment Setup

### 1. Create Virtual Environment
```bash
uv venv --clear
uv sync --all-groups
```

### 2. Verify Installation
```bash
# Check that ml_playground is importable
uv run python -c "import ml_playground; print('✓ Setup complete')"
```

## Basic Workflow Commands

### Dataset Preparation
```bash
uv run python -m ml_playground.cli prepare shakespeare
uv run python -m ml_playground.cli prepare bundestag_char
```

### Training
```bash
uv run python -m ml_playground.cli train ml_playground/configs/shakespeare_cpu.toml
uv run python -m ml_playground.cli train ml_playground/configs/bundestag_char_cpu.toml
```

### Sampling
```bash
uv run python -m ml_playground.cli sample ml_playground/configs/shakespeare_cpu.toml
uv run python -m ml_playground.cli sample ml_playground/configs/bundestag_char_cpu.toml
```

### End-to-End Loop
```bash
uv run python -m ml_playground.cli loop bundestag_char ml_playground/configs/bundestag_char_cpu.toml
```

## Configuration System

- All configuration via TOML files mapped to dataclasses in `ml_playground/config.py`
- No CLI parameter overrides - TOML is the single source of truth
- Paths automatically converted to `pathlib.Path` objects
- Device defaults to CPU-first; MPS/CUDA supported when explicitly configured

## Quick Troubleshooting

**Tests cannot import `ml_playground`**: You're not in the project venv - run `uv venv` then `uv sync --all-groups`

**Missing pytest**: Run `uv sync --all-groups` to install dev dependencies

**UV hangs during venv creation**: Exit any active virtual environment first, then retry

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
ml-playground: strict, typed, UV-only training/sampling module

This module provides a single, one-way interface to prepare data, train, and sample.
It is CPU/MPS-friendly, strictly typed, and uses TOML configs.

Policy
- UV is mandatory for all workflows (venv, dependency sync, running tools). Do not use pip, requirements.txt, or uvx.
- Never set PYTHONPATH. Running inside the project, venv ensures ml_playground is importable.
- Quality tooling is mandatory before commit (ruff, mypy, pyright), and tests must pass.

Prerequisites
- Install UV: https://docs.astral.sh/uv/

Setup (required)
- Create a venv and sync all dependency groups (runtime + dev):
  uv venv
  uv sync --all-groups

Quality gates (required before commit/PR)
- Lint/format/imports:
  uv run ruff check --fix . && uv run ruff format .
- Static analysis and typing:
  uv run pyright
  uv run mypy ml_playground
- Tests:
  uv run pytest -n auto -W error --strict-markers --strict-config -v

Datasets
- Shakespeare (GPT-2 BPE; prepared via internal ml_playground.datasets.shakespeare)
- Bundestag (char-level; prepared via internal ml_playground.datasets.bundestag_char; auto-seeds data/bundestag_char/page1.txt from a bundled sample if missing — replace it with your own text for real runs)

Prepare
- Shakespeare:
  uv run python -m ml_playground.cli prepare shakespeare

- Bundestag (char-level):
  uv run python -m ml_playground.cli prepare bundestag_char

Train
- Shakespeare (example config):
  uv run python -m ml_playground.cli train ml_playground/configs/shakespeare_cpu.toml

- Bundestag (char-level example):
  uv run python -m ml_playground.cli train ml_playground/configs/bundestag_char_cpu.toml

Sample
- Using the same TOML (sampler tries ckpt_best.pt, then ckpt_last.pt, then legacy ckpt.pt in out_dir):
  uv run python -m ml_playground.cli sample ml_playground/configs/shakespeare_cpu.toml
  uv run python -m ml_playground.cli sample ml_playground/configs/bundestag_char_cpu.toml

Notes
- Configuration is done strictly via TOML dataclasses (see ml_playground/config.py). No CLI overrides.
- CPU/MPS are first-class. CUDA can be selected in the TOML if available.
- Checkpoints: trainer writes ckpt_last.pt on every eval and updates ckpt_best.pt when improved (or when always_save_checkpoint is true). Training auto-resumes from ckpt_last.pt if it exists; to start fresh, delete ckpt_last.pt (and ckpt_best.pt optionally) or use a new out_dir. On resume, the checkpointed model_args (n_layer, n_head, n_embd, block_size, bias, vocab_size, dropout) take precedence over TOML values to ensure compatibility.
- For small local runs, tune batch_size, block_size, and grad_accum_steps in the [train.data] section.

Loop
- End-to-end in one command (bundestag_char):
  uv run python -m ml_playground.cli loop bundestag_char ml_playground/configs/bundestag_char_cpu.toml

- Shakespeare end-to-end:
  uv run python -m ml_playground.cli loop shakespeare ml_playground/configs/shakespeare_cpu.toml

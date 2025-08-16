_next: strict, typed, uv-first training/sampling module

This folder provides a single, “one way only” interface to prepare data, train, and sample.
It is CPU/MPS-friendly, strictly typed, and uses TOML configs.

We use uv for everything (virtualenv, dependency sync, running tools).

Prerequisites
- Install uv: https://docs.astral.sh/uv/

Setup
- Create & activate a venv and sync dependencies from pyproject.toml:
  uv venv
  uv sync

- Run tests to verify:
  uvx pytest -q _next/tests
  # or if added to project deps: uv run pytest -q _next/tests

Datasets
- Shakespeare (GPT-2 BPE; prepared via internal _next.datasets.shakespeare)
- Bundestag (char-level; prepared via internal _next.datasets.bundestag_char; auto-seeds data/bundestag_char/page1.txt from a bundled sample if missing — replace it with your own text for real runs)

Prepare
- Shakespeare:
  uv run python -m _next.cli prepare shakespeare

- Bundestag (char-level):
  uv run python -m _next.cli prepare bundestag_char

Train
- Shakespeare (example config):
  uv run python -m _next.cli train _next/configs/shakespeare_cpu.toml

- Bundestag (char-level example):
  uv run python -m _next.cli train _next/configs/bundestag_char_cpu.toml

Sample
- Using the same TOML (sampler tries ckpt_best.pt, then ckpt_last.pt, then legacy ckpt.pt in out_dir):
  uv run python -m _next.cli sample _next/configs/shakespeare_cpu.toml
  uv run python -m _next.cli sample _next/configs/bundestag_char_cpu.toml

Notes
- Configuration is strictly via TOML dataclasses (see _next/config.py). No CLI overrides.
- CPU/MPS are first-class. CUDA can be selected in the TOML if available.
- Checkpoints: trainer writes ckpt_last.pt on every eval and updates ckpt_best.pt when improved (or when always_save_checkpoint is true). Training auto-resumes from ckpt_last.pt if it exists; to start fresh, delete ckpt_last.pt (and ckpt_best.pt optionally) or use a new out_dir. On resume, the checkpointed model_args (n_layer, n_head, n_embd, block_size, bias, vocab_size, dropout) take precedence over TOML values to ensure compatibility.
- For small local runs, tune batch_size, block_size, and grad_accum_steps in the [train.data] section.


Loop
- End-to-end in one command (bundestag_char):
  uv run python -m _next.cli loop bundestag_char _next/configs/bundestag_char_cpu.toml

- (Optional) Shakespeare end-to-end:
  uv run python -m _next.cli loop shakespeare _next/configs/shakespeare_cpu.toml

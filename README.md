ml-playground: strict, typed, UV-only training/sampling module

This module provides a single, one-way interface to prepare data, train, and sample.
It is CPU/MPS-friendly, strictly typed, and uses TOML configs.

Policy
- UV is mandatory for all workflows (venv, dependency sync, running tools). Do not use pip or requirements.txt.
- Prefer Makefile targets for common workflows. Under the hood, they call `uv run ...` consistently.
- Never set PYTHONPATH. Running inside the project venv ensures `ml_playground` is importable.
- Quality tooling is mandatory before commit (ruff, mypy, pyright), and tests must pass.
- Linear history for own work: rebase your branches and avoid merge commits; fast-forward only. See DEVELOPMENT.md → “Git Workflow: Linear history”.

Prerequisites
- Install UV: https://docs.astral.sh/uv/

Setup (required)
- Create a venv and sync all dependency groups (runtime + dev):
  make setup

- Alternatively (manual):
  uv venv --clear && uv sync --all-groups

Quality gates (required before commit/PR)
- Lint/format/imports:
  make format
- Static analysis and typing:
  make pyright
  make mypy
- Full quality gate:
  make quality
- Tests:
  make test

Datasets
- Shakespeare (GPT-2 BPE; prepared via internal ml_playground.experiments.shakespeare)
 - Bundestag (char-level; prepared via internal ml_playground.experiments.bundestag_char; requires a user-provided text at ml_playground/experiments/bundestag_char/datasets/input.txt)
- Bundestag (tiktoken BPE; prepared via internal ml_playground.experiments.bundestag_tiktoken)

Prepare
- Shakespeare:
  uv run python -m ml_playground.cli prepare shakespeare

- Bundestag (char-level):
  uv run python -m ml_playground.cli prepare bundestag_char

- Bundestag (tiktoken BPE):
  uv run python -m ml_playground.cli prepare bundestag_tiktoken

Train
- Shakespeare:
  uv run python -m ml_playground.cli train shakespeare --exp-config ml_playground/configs/shakespeare_cpu.toml

- Bundestag (char-level):
  uv run python -m ml_playground.cli train bundestag_char --exp-config ml_playground/configs/bundestag_char_cpu.toml

Sample
- Using the experiment's config.toml (sampler tries ckpt_best.pt, then ckpt_last.pt, then legacy ckpt.pt in out_dir):
  uv run python -m ml_playground.cli sample shakespeare --exp-config ml_playground/configs/shakespeare_cpu.toml
  uv run python -m ml_playground.cli sample bundestag_char --exp-config ml_playground/configs/bundestag_char_cpu.toml

Notes
- Dataset preparers are registered from ml_playground/experiments and the CLI discovers them automatically. The ml_playground/datasets package is optional and may be absent.
- Configuration is done strictly via TOML dataclasses (see ml_playground/config.py). No CLI overrides.
- CPU/MPS are first-class. CUDA can be selected in the TOML if available.
- Checkpoints: trainer writes ckpt_last.pt on every eval and updates ckpt_best.pt when improved (or when always_save_checkpoint is true). Training auto-resumes from ckpt_last.pt if it exists; to start fresh, delete ckpt_last.pt (and ckpt_best.pt optionally) or use a new out_dir. On resume, the checkpointed model_args (n_layer, n_head, n_embd, block_size, bias, vocab_size, dropout) take precedence over TOML values to ensure compatibility.
- For small local runs, tune batch_size, block_size, and grad_accum_steps in the [train.data] section.
- For detailed information about the centralized framework utilities, see [Framework Utilities Documentation](docs/framework_utilities.md).

Loop
- End-to-end in one command (bundestag_char):
  uv run python -m ml_playground.cli loop bundestag_char --exp-config ml_playground/configs/bundestag_char_cpu.toml

- Shakespeare end-to-end:
  uv run python -m ml_playground.cli loop shakespeare --exp-config ml_playground/configs/shakespeare_cpu.toml


TensorBoard (auto-enabled)
- Training automatically logs to TensorBoard for both the generic trainer and the HF+PEFT finetuning integration (no config flags needed).
- Log directory: out_dir/logs/tb inside your configured out_dir.
- Scalars:
  - train/loss, val/loss
  - train/lr
  - train/tokens_per_sec
  - train/step_time_ms (generic trainer only)
- View the dashboard:
  uv run tensorboard --logdir out/<your_out_dir>/logs/tb --port 6006
  Then open http://localhost:6006

GGUF export (vendor approach)
- Place llama.cpp’s converter at a stable path in this repo:
  tools/llama_cpp/convert-hf-to-gguf.py
- Copy the upstream file from:
  https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py
- Configure your exporter to use it by setting [export.ollama].convert_bin to the absolute path above.
- Verify:
  uv run python tools/llama_cpp/convert-hf-to-gguf.py --help
  (If you see a placeholder message, you still need to copy the upstream script.)

Testing
- Layout
  - Unit: `tests/unit/`
  - Integration: `tests/integration/`
  - E2E: `tests/e2e/`
  - Acceptance: `tests/acceptance/`
- Markers (auto-applied by per-folder conftest)
  - `unit` (implicit for unit folder)
  - `integration`
  - `e2e`
  - `acceptance`
- Run examples
  - All tests: `make test` (or `uv run pytest -n auto -W error --strict-markers --strict-config -v`)
  - Unit only: `make unit` (or `uv run pytest tests/unit -n auto -W error --strict-markers --strict-config -v`)
  - Integration only: `uv run pytest -m integration -n auto -W error --strict-markers --strict-config -v`
  - E2E only: `uv run pytest -m e2e -n auto -W error --strict-markers --strict-config -v`
  - Acceptance only: `uv run pytest -m acceptance -q`
  
See `tests/unit/README.md`, `tests/integration/README.md`, and `tests/e2e/README.md` for scope and patterns.

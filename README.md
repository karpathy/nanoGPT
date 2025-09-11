# ml-playground: strict, typed, UV-only training/sampling module

This module provides a single, one-way interface to prepare data, train, and sample.
It is CPU/MPS-friendly, strictly typed, and uses TOML configs.

## Documentation abstraction policy

- Top-level docs are high-level and describe the why and the overall layout.
- Each subfolder contains its own `Readme.md` with a focused scope and a folder tree.
- The deeper you go in the directory tree, the lower the level of abstraction and the more operational details you’ll find.

## Repository structure (high-level)

```text
.
├── ml_playground/             - core module (configs, experiments, runtime code)
│   ├── analysis/              - analysis tools (e.g., LIT integration)
│   ├── datasets/              - optional package; experiments can run without it
│   ├── experiments/           - self-contained experiments (mid-level docs)
│   └── configs/               - example configs referenced by README/Makefile
├── tests/                     - test suite (see per-folder README for scope)
│   ├── unit/                  - low-level API tests
│   ├── integration/           - multi-module tests via Python APIs
│   ├── e2e/                   - CLI-level smoke tests
│   └── acceptance/            - higher-level behaviors and policies
├── tools/                     - developer tools and vendor integrations
│   └── llama_cpp/             - GGUF conversion helper (vendored instructions)
├── docs/                      - supplementary docs (framework utilities, LIT, etc.)
├── lit_nlp/                   - optional LIT integration
├── Makefile                   - entrypoints for setup, quality gates, runtime
├── pyproject.toml             - strict typing/linting/testing configuration
└── README.md                  - this file (top-level, high abstraction)
```

## Policy

- Use Make targets for all workflows (env setup, quality, tests, runtime). Under the hood, they run via uv.
- Never set PYTHONPATH. Running inside the project venv ensures `ml_playground` is importable.
- Quality tooling is mandatory before commit (ruff, mypy, pyright), and tests must pass.
- Linear history for own work: rebase your branches and avoid merge commits; fast-forward only. See DEVELOPMENT.md → “Git Workflow: Linear history”.

Prerequisites

- Install UV: <https://docs.astral.sh/uv/>

Setup (required)

- Create a venv and sync all dependency groups (runtime + dev):
  make setup
  make verify

Quality gates (required before commit/PR)

- Lint/format/imports:
  make format
- Static analysis and typing:
  make pyright
  make mypy
- Full quality gate (ruff, format, pyright, mypy, pytest):
  make quality
  
- Extended quality (optional, non-fatal mutation testing with Cosmic Ray):
  make quality-ext
- Tests:
  make test

Datasets

- Shakespeare (GPT-2 BPE; prepared via internal ml_playground.experiments.shakespeare)
- Bundestag (char-level; prepared via internal ml_playground.experiments.bundestag_char; requires a user-provided text at ml_playground/experiments/bundestag_char/datasets/input.txt)
- Bundestag (tiktoken BPE; prepared via internal ml_playground.experiments.bundestag_tiktoken)

Prepare

- Shakespeare:
  make prepare EXP=shakespeare

- Bundestag (char-level):
  make prepare EXP=bundestag_char

- Bundestag (tiktoken BPE):
  make prepare EXP=bundestag_tiktoken

Train

- Shakespeare:
  make train EXP=shakespeare CONFIG=ml_playground/configs/shakespeare_cpu.toml

- Bundestag (char-level):
  make train EXP=bundestag_char CONFIG=ml_playground/configs/bundestag_char_cpu.toml

Sample

- Using the experiment's config.toml (sampler tries ckpt_best.pt, then ckpt_last.pt, then legacy ckpt.pt in out_dir):
  make sample EXP=shakespeare CONFIG=ml_playground/configs/shakespeare_cpu.toml
  make sample EXP=bundestag_char CONFIG=ml_playground/configs/bundestag_char_cpu.toml

Notes

- Dataset preparers are registered from ml_playground/experiments and the CLI discovers them automatically. The ml_playground/datasets package is optional and may be absent.
- Configuration is done strictly via TOML dataclasses (see ml_playground/config.py). No CLI overrides.
- CPU/MPS are first-class. CUDA can be selected in the TOML if available.
- Checkpoints: trainer writes ckpt_last.pt on every eval and updates ckpt_best.pt when improved (or when always_save_checkpoint is true). Training auto-resumes from ckpt_last.pt if it exists; to start fresh, delete ckpt_last.pt (and ckpt_best.pt optionally) or use a new out_dir. On resume, the checkpointed model_args (n_layer, n_head, n_embd, block_size, bias, vocab_size, dropout) take precedence over TOML values to ensure compatibility.
- For small local runs, tune batch_size, block_size, and grad_accum_steps in the [train.data] section.
- For detailed information about the centralized framework utilities, see [Framework Utilities Documentation](docs/framework_utilities.md).

Mutation testing

- Mutation testing is performed with Cosmic Ray and configured centrally in `pyproject.toml` (see `[tool.cosmic-ray]`).
- Run manually when needed (non-fatal in Makefile):
  - `make quality-ext`
- The Makefile initializes a session database at `out/cosmic-ray/session.sqlite` if absent and then runs `cosmic-ray exec` against it.
- CI and pre-commit do not run mutation testing by default.

Loop

- End-to-end in one command (bundestag_char):
  make loop EXP=bundestag_char CONFIG=ml_playground/configs/bundestag_char_cpu.toml

- Shakespeare end-to-end:
  make loop EXP=shakespeare CONFIG=ml_playground/configs/shakespeare_cpu.toml

TensorBoard (auto-enabled)

- Training automatically logs to TensorBoard for both the generic trainer and the HF+PEFT finetuning integration (no config flags needed).
- Log directory: out_dir/logs/tb inside your configured out_dir.
- Scalars:
  - train/loss, val/loss
  - train/lr
  - train/tokens_per_sec
  - train/step_time_ms (generic trainer only)
- View the dashboard:
  make tensorboard LOGDIR=out/<your_out_dir>/logs/tb [PORT=6006]
  Then open <http://localhost:6006>

GGUF export (vendor approach)

- Place llama.cpp’s converter at a stable path in this repo:
  tools/llama_cpp/convert-hf-to-gguf.py
- Copy the upstream file from:
  <https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert-hf-to-gguf.py>
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
  - All tests: `make test`
  - Unit only: `make unit`
  - Unit w/ coverage: `make unit-cov`
  - Integration only: `make integration`
  - E2E only: `make e2e`
  - Acceptance only: `make acceptance`
  
See `tests/unit/README.md`, `tests/integration/README.md`, and `tests/e2e/README.md` for scope and patterns.

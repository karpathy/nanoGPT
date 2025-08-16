# _next Developer Guidelines (STRICT UV-ONLY)

Audience: Advanced contributors working exclusively on the `_next` module. These rules are binding. Follow them exactly.

Key policies (non-negotiable)
- Use UV for everything: virtualenv, dependency sync, running tools and tests.
- Do not use pip, requirements.txt, uvx, or raw venv activation commands.
- Never set PYTHONPATH. Running inside the project venv ensures `_next` is importable.
- Quality tooling is mandatory. Linting, formatting, static analysis, and typing checks must pass before any commit/PR.


## Environment & Configuration (required)
- Python version: see pyproject.toml (currently ">=3.9,<3.13").
- Create venv and sync all dependency groups (runtime + dev):
  - uv venv
  - uv sync --all-groups

Runtime entry points (no build step)
- Prepare datasets:
  - uv run python -m _next.cli prepare shakespeare
  - uv run python -m _next.cli prepare bundestag_char
- Train from TOML:
  - uv run python -m _next.cli train _next/configs/shakespeare_cpu.toml
  - uv run python -m _next.cli train _next/configs/bundestag_char_cpu.toml
- Sample from TOML (tries ckpt_best.pt, then ckpt_last.pt, then legacy ckpt.pt in out_dir):
  - uv run python -m _next.cli sample _next/configs/shakespeare_cpu.toml
  - uv run python -m _next.cli sample _next/configs/bundestag_char_cpu.toml
- End-to-end loop (prepare → train → sample):
  - uv run python -m _next.cli loop bundestag_char _next/configs/bundestag_char_cpu.toml

Configuration model (TOML-only)
- All configuration is via TOML mapped to dataclasses in _next/config.py. No CLI overrides.
- Paths in TOML are coerced to pathlib.Path where applicable (e.g., dataset_dir, out_dir).
- Checkpointing and resume (RuntimeConfig): trainer writes ckpt_last.pt every eval; updates ckpt_best.pt on improvement (or when always_save_checkpoint=true). On resume, checkpointed model_args (n_layer, n_head, n_embd, block_size, bias, vocab_size, dropout) override TOML for compatibility.
- Device defaults are CPU-first; MPS/CUDA are supported if explicitly selected in TOML.


## Mandatory Quality Gates (run before every commit/PR)
- Lint, format, imports:
  - uv run ruff check --fix . && uv run ruff format .
- Static analysis:
  - uv run pyright
- Type checking:
  - uv run mypy _next
- Tests:
  - uv run python -m pytest -q

All four gates must pass. Do not open a PR otherwise.


## Testing Workflow (authoritative)
Baseline: After `uv sync --all-groups`, runtime deps (numpy, torch, etc.) and dev tools (pytest, ruff, mypy, pyright) are available in the project venv.

Run tests from project root only:
- Full suite:
  - uv run pytest -q
- Filtered (example):
  - uv run pytest -q -k "config or data"

Adding a new test
- Create a file under `_next/tests/test_<name>.py`.
- Example (stdlib-only):

  from pathlib import Path
  from _next.config import load_toml, AppConfig

  def test_guidelines_demo(tmp_path: Path) -> None:
      toml_text = """
  [train.model]
  n_layer=1
  n_head=1
  n_embd=32
  block_size=16
  bias=false

  [train.data]
  dataset_dir = "data/shakespeare"
  block_size = 16
  batch_size = 2
  grad_accum_steps = 1

  [train.optim]
  learning_rate = 0.001

  [train.schedule]

  [train.runtime]
  out_dir = "out/test_next"
  max_iters = 1

  [sample.runtime]
  out_dir = "out/test_next"

  [sample.sample]
  """
      cfg_path = tmp_path / "cfg.toml"
      cfg_path.write_text(toml_text)

      cfg: AppConfig = load_toml(cfg_path)
      assert cfg.train is not None
      assert cfg.sample is not None

- Execute:
  - uv run pytest -q _next/tests/test_guidelines_demo.py
- Remove demo artifacts after verification.


## Additional Development Information
- Code style and typing:
  - Code is strictly typed; dataclasses define config schemas; use explicit types and pathlib.Path for filesystem paths.
  - Keep CLI free of config mutation logic; TOML is the single source of truth.
  - Favor pure functions for data preparation; make device selection explicit.
- Checkpointing and resume:
  - On resume, checkpointed model_args override TOML to ensure shape compatibility. If shapes must change, start with a fresh out_dir or delete ckpt_last.pt/ckpt_best.pt.
- Dataset notes:
  - The char-level Bundestag dataset autoseeds data from a bundled sample resource; replace it with real data for non-trivial runs.
- CPU/MPS/CUDA:
  - Use device="cpu" for tests and local CI; MPS/CUDA may be used when explicitly configured and available.


## Troubleshooting (strict)
- Tests cannot import `_next`:
  - You are not running inside the project venv. Run `uv venv` then `uv sync --all-groups` and execute commands with `uv run` from the project root.
- `uv run pytest` fails due to missing pytest:
  - You did not sync dev tools. Run `uv sync --all-groups`.
- Torch wheels:
  - Use a supported Python version (see pyproject). Run CPU-first configurations in tests.


## TL;DR (do this, in order)
- uv venv
- uv sync --all-groups
- uv run ruff check --fix . && uv run ruff format .
- uv run pyright
- uv run mypy _next
- uv run pytest -q _next/tests

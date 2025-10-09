# tools/

Centralized developer utilities and helper scripts that support the `ml_playground` project. These scripts complement the
three Typer CLIs published in `pyproject.toml`:

- `env-tasks` – environment setup, lint/type checks, cache cleanup, TensorBoard/LIT helpers.
- `test-tasks` – pytest orchestration for the various suites and coverage helpers.
- `ci-tasks` – quality gates, coverage generation, and Cosmic Ray automation used by GitHub Actions.

## Purpose

- Provide small, focused utilities used during development and maintenance.
- Keep operational scripts discoverable and documented in one place.
- Avoid raw `pip` or manual virtualenv activation by standardizing on `uv`/`uvx`.

## Structure

- `env_tasks.py` — developer environment commands (lint, typecheck, caches, TensorBoard, LIT).
- `test_tasks.py` — pytest entry points for each suite.
- `ci_tasks.py` — CI-focused flows (quality, coverage, mutation).
- `task_utils.py` — shared helpers consumed by the CLIs above.
- `port_kill.py` — kill a process bound to a TCP port (Mac/Linux).
- `cleanup_ignored_tracked.py` — remove accidentally tracked files that should be ignored.
- `setup_ai_guidelines.py` — configure symlinks for AI pair-programming workflows.
- `mutation_summary.py` / `mutation_report.py` — summarize Cosmic Ray state alongside `uvx --from . ci-tasks mutation run`.
- `llama_cpp/` — vendor instructions and helpers for GGUF conversion.

## Usage

Always run through the project venv using UV. From repo root:

```bash
# Environment lifecycle
uvx --from . env-tasks setup
uvx --from . env-tasks verify

# Linting / formatting / type checks
uvx --from . env-tasks lint
uvx --from . env-tasks typecheck

# Test suites
uvx --from . test-tasks unit
uvx --from . test-tasks integration -- -k "experiment"

# CI parity
uvx --from . ci-tasks quality
uvx --from . ci-tasks coverage-report --fail-under 87

# Utility scripts remain available via uv run
uv run python tools/cleanup_ignored_tracked.py --dry-run
uv run python tools/port_kill.py 6006
```

## Examples

- TensorBoard port is stuck on 6006:

```bash
uv run python tools/port_kill.py 6006
uvx --from . env-tasks tensorboard --logdir out/<run>/logs/tb
```

- Clean up noisy artifacts that slipped into Git:

```bash
uv run python tools/cleanup_ignored_tracked.py --dry-run
uv run python tools/cleanup_ignored_tracked.py --apply
```

- Refresh mutation artifacts locally:

```bash
uvx --from . ci-tasks mutation run
uvx --from . ci-tasks coverage-report --verbose
```

## Conventions

- UV-only: invoke tools with `uvx` or `uv run python ...` to use the project environment.
- Keep scripts self-contained, documented, and under 200 LOC where practical.
- Prefer clear CLI flags and `--help` text; avoid hidden behavior.
- Align documentation with `.dev-guidelines/DOCUMENTATION.md` when editing this file or adding tool docs; keep mutation workflow notes in `.dev-guidelines/TESTING.md`.

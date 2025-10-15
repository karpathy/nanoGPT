# tools/

Centralized developer utilities and helper scripts that support the `ml_playground` project. Everything
is invoked via UV CLIs—no raw `pip`, no manual venv activation.

## Purpose

- Provide small, focused utilities used during development and maintenance.
- Keep operational scripts discoverable and documented in one place.

## Structure

- `ci_tasks.py` — Typer CLI exposing quality gates (`uv run ci-tasks quality`), coverage workflows, and mutation helpers.
- `env_tasks.py` — Typer CLI for environment setup, verification, cache cleanup, TensorBoard, and AI guideline symlinks (`uv run env-tasks <command>`).
- `lint_tasks.py` — Typer CLI bundling lint/format slices for fast feedback (`uv run lint-tasks <command>`).
- `lit_tasks.py` — Typer CLI for LIT integration helpers (`uv run lit-tasks <command>`).
- `test_tasks.py` — Typer CLI orchestrating pytest suites (`uv run test-tasks <suite>`).
- `task_utils.py` — shared helpers (UV process wrappers, cache helpers) used by the CLIs above.
- `cleanup_ignored_tracked.py` — remove accidentally tracked files that should be ignored.
- `mutation_summary.py` — prints the active Cosmic Ray configuration before mutation runs.
- `mutation_report.py` — summarizes mutant outcomes after a Cosmic Ray run.
- `port_kill.py` — kill a process bound to a TCP port (Mac/Linux).
- `setup_ai_guidelines.py` — configure symlinks for AI pair-programming workflow per guideline docs.
- `llama_cpp/` — vendor instructions and helpers for GGUF conversion.

## Usage

Always run through the project venv using UV. From repo root:

```bash
# Quality gates
uv run ci-tasks quality

# Run GitHub quality workflow locally via act
uv run ci-tasks quality-ci-local

# Coverage report with threshold enforcement
uv run ci-tasks coverage-report --fail-under 87

# Run unit tests
uv run test-tasks unit

# Fast lint bundle
uv run lint-tasks ruff

# Environment setup
uv run env-tasks setup
```

- **`quality-ci-local`**: Binds `.cache/uv`, `.cache/pre-commit`, `.cache/ruff`, and `.venv` into the container. Toggle mounts with `--no-bind-caches` or pass additional flags directly to `act`.

## Examples

- TensorBoard port is stuck on 6006:

```bash
uv run python tools/port_kill.py 6006
uv run env-tasks tensorboard --logdir out/<run>/logs/tb
```

- Clean up noisy artifacts that slipped into Git:

```bash
uv run python tools/cleanup_ignored_tracked.py --dry-run
uv run python tools/cleanup_ignored_tracked.py --apply
```

## Conventions

- UV-only: invoke tools with `uv run python ...` to use the project environment.
- Keep scripts self-contained, documented, and under 200 LOC where practical.
- Prefer clear CLI flags and `--help` text; avoid hidden behavior.
- Align documentation with `.dev-guidelines/DOCUMENTATION.md` when editing this file or adding tool docs; keep mutation workflow notes in `.dev-guidelines/TESTING.md`.

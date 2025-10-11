# tools/

Centralized developer utilities and helper scripts that support the `ml_playground` project. Everything
is invoked via UV CLIs—no raw `pip`, no manual venv activation.

## Purpose

- Provide small, focused utilities used during development and maintenance.
- Keep operational scripts discoverable and documented in one place.

## Structure

- `ci_tasks.py` — Typer CLI exposing quality gates (`uvx --from . ci-tasks quality`), coverage workflows, and mutation helpers.
- `env_tasks.py` — Typer CLI for environment setup, verification, cache cleanup, TensorBoard, and AI guideline symlinks (`uvx --from . env-tasks <command>`).
- `lint_tasks.py` — Typer CLI bundling lint/format slices for fast feedback (`uvx --from . lint-tasks <command>`).
- `lit_tasks.py` — Typer CLI for LIT integration helpers (`uvx --from . lit-tasks <command>`).
- `test_tasks.py` — Typer CLI orchestrating pytest suites (`uvx --from . test-tasks <suite>`).
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
uvx --from . ci-tasks quality

# Coverage report with threshold enforcement
uvx --from . ci-tasks coverage-report --fail-under 87

# Run unit tests
uvx --from . test-tasks unit

# Fast lint bundle
uvx --from . lint-tasks ruff

# Environment setup
uvx --from . env-tasks setup
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

## Conventions

- UV-only: invoke tools with `uv run python ...` to use the project environment.
- Keep scripts self-contained, documented, and under 200 LOC where practical.
- Prefer clear CLI flags and `--help` text; avoid hidden behavior.
- Align documentation with `.dev-guidelines/DOCUMENTATION.md` when editing this file or adding tool docs; keep mutation workflow notes in `.dev-guidelines/TESTING.md`.

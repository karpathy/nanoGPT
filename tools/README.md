# tools/

<details>
<summary>Related documentation</summary>

- [Documentation Guidelines](../.dev-guidelines/DOCUMENTATION.md) – Unified standards for all repository docs, covering top-level, module, experiment, test, and tool content.
- [Development Practices](../.dev-guidelines/DEVELOPMENT.md) – Core development workflow, quality gates, and commit standards for contributors.
- [Testing Standards](../.dev-guidelines/TESTING.md) – Strict TDD workflow and mandatory testing policy that tool-assisted workflows must respect.
- [Top-level README](../README.md) – High-level orientation to repository structure and entry points, including the tools directory.

</details>

Centralized developer utilities and helper scripts that support the `ml_playground` project. These are optional, mostly
used for convenience during development. (No raw pip, no manual venv activation).

## Purpose

- Provide small, focused utilities used during development and maintenance.
- Keep operational scripts discoverable and documented in one place.

## Structure

- `port_kill.py` — kill a process bound to a TCP port (Mac/Linux).
- `cleanup_ignored_tracked.py` — remove accidentally tracked files that should be ignored.
- `setup_ai_guidelines.py`: Configures symlinks for AI pair-programming workflow per guideline docs.
- `mutation_summary.py` — prints the active Cosmic Ray configuration before `make mutation`.
- `mutation_report.py` — summarizes mutant outcomes after a Cosmic Ray run.
- `llama_cpp/` — vendor instructions and helpers for GGUF conversion.

## Usage

Always run through the project venv using UV. From repo root:

```bash
# List available tools (this file)
open tools/README.md

# Kill process bound to port 6006 (TensorBoard)
uv run python tools/port_kill.py 6006

# Clean accidentally tracked ignored files (dry-run first)
uv run python tools/cleanup_ignored_tracked.py --dry-run
uv run python tools/cleanup_ignored_tracked.py --apply

# (Optional) LLaMA GGUF conversion steps
# See vendor README inside the directory for exact commands.
open tools/llama_cpp/README.md

# Mutation summary (pre-run expectations)
uv run python tools/mutation_summary.py --config pyproject.toml

# Mutation report (post-run survivor counts)
uv run python tools/mutation_report.py --config pyproject.toml
```

## Examples

- TensorBoard port is stuck on 6006:

```bash
uv run python tools/port_kill.py 6006
make tensorboard LOGDIR=out/<run>/logs/tb
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

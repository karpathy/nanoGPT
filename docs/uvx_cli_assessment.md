# Assessment: Replacing Makefile with `dev-tasks` UVX CLI

This document captures the evaluation of the recent workflow change that introduced the Typer-powered `tools/dev_tasks.py` script as the canonical developer interface while retaining a thin top-level `Makefile` that delegates to the CLI.

## Pros

- **Single, cross-platform entry point** – `tools/dev_tasks.py` provides all developer workflows through uv-managed execution, avoiding GNU Make requirements on Windows environments while keeping the commands consistent across platforms.【F:tools/dev_tasks.py†L1-L118】
- **Self-documented UX** – Typer automatically renders `--help` output and subcommands, reducing the need to search for hidden Make targets and aligning with the structure promoted in the README.【F:tools/dev_tasks.py†L28-L41】【F:README.md†L42-L58】
- **Better dependency hygiene** – The CLI shells out through `uv` for every task, ensuring the same resolver that installs dependencies also runs quality gates and tests without leaking host environments.【F:tools/dev_tasks.py†L44-L93】

## Cons

- **Loss of ubiquitous Make idioms** – Many contributors expect `make test`/`make lint` muscle memory; the new workflow requires relearning `uvx --from . dev-tasks …` invocations, though thin Makefile wrappers now map the legacy targets to the new commands to ease the transition.【F:Makefile†L1-L164】【F:README.md†L42-L58】
- **Python-only bootstrap** – Running developer tooling now depends on a working Python interpreter capable of executing the Typer script, whereas Make could orchestrate non-Python tooling (e.g., container builds) without that prerequisite.【F:tools/dev_tasks.py†L1-L118】
- **Script complexity vs. declarative recipes** – Extending behaviors now requires editing Python functions with subprocess handling instead of adding declarative Make targets, increasing maintenance burden and the risk of runtime errors in the command wrapper.【F:tools/dev_tasks.py†L47-L118】

## Recommendation

Retain the `dev-tasks` CLI as the canonical interface for uv-managed Python workflows, paired with the thin Makefile wrappers that map common targets (e.g., `make test`, `make lint`) to the new commands. This preserves developer familiarity while keeping the uv-first execution model.【F:Makefile†L1-L164】【F:tools/dev_tasks.py†L84-L118】【F:README.md†L42-L58】

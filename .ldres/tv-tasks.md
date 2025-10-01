# tv-tasks

This file tracks targeted refactoring tasks Thomas requested. Keep items small, reviewable, and enforce our UV + pre-commit workflow. Each task includes scope, acceptance criteria, and commit guidance.

---

## Task 1: Reduce Makefile indirections in test targets

- Scope:
  - In `Makefile`, replace indirections like `$(MAKE) pytest-core PYARGS="-m integration --no-cov"` with direct invocations using `$(RUN) pytest` and `$(PYTEST_BASE)`.
  - Targets to simplify: `integration`, `e2e`, and any others using `pytest-core` where a single direct call is clearer.
- Acceptance criteria:
  - `make integration` and `make e2e` call `$(RUN) pytest $(PYTEST_BASE) ...` directly.
  - `make test`, `make pytest-all`, and pre-commit continue to pass locally and in CI.
- Commit guidance:
  - Branch: `chore/makefile-simplify-test-targets`
  - Commit: `chore(makefile): reduce test target indirections; invoke pytest directly`

---

## Task 2: Integrate standalone units into canonical packages

- Target units:
  - `ml_playground/checkpoint.py`
  - `ml_playground/ema.py`
  - `ml_playground/estimator.py`
  - `ml_playground/lr_scheduler.py`
  - `ml_playground/tokenizer_protocol.py`
- Scope:
  - Move or refactor these modules under their canonical package locations (`ml_playground/training/checkpointing/`, `ml_playground/training/`, `ml_playground/models/utils/` or `core/`, `ml_playground/training/optim/`, `ml_playground/core/`), aligned with prior restructurings.
  - Update all imports across the codebase and tests to the new canonical paths.
  - Ensure there are no re-export facades; comply with `IMPORT_GUIDELINES.md`.
- Acceptance criteria:
  - All imports reference canonical modules; no legacy paths remain.
  - `make quality` and the full test suite pass.
  - No circular imports introduced; `pyright` clean.
- Commit guidance:
  - Use small commits, one module at a time:
    - `refactor(training): move checkpoint helpers to training/checkpointing`
    - `refactor(training): move ema to training/ema`
    - `refactor(models): move estimator to models/utils` (or agreed location)
    - `refactor(training): move lr_scheduler to training/optim`
    - `refactor(core): move tokenizer_protocol to core/`
  - Follow up with `docs(...)` if READMEs or guides reference old paths.

---

## Task 3: Rename refactorings list to tv-tasks.md

- Scope:
  - Adopt `.ldres/tv-tasks.md` as the source of truth for Thomas's task list.
  - Optionally add a pointer note at the top of `.ldres/Refactorings.md` indicating this file supersedes it.
- Acceptance criteria:
  - `.ldres/tv-tasks.md` exists and is referenced in ongoing work.
  - Team uses this file for new action items.
- Commit guidance:
  - Branch: `docs/tv-tasks`
  - Commit: `docs(tasks): add tv-tasks.md and point from Refactorings.md`

---

## Notes

- `make quality` is the canonical gate and already runs pre-commit hooks (ruff, format, checkmake, pyright, mypy, vulture, pytest). No separate test run in the Make target is necessary.
- When the "remove mocks" effort is complete, enforce no mock usage via Ruff banned-modules and a CI grep sweep.

---

## Task 4: Rename module/project to ml_playgound (spelling per request)

- Scope:
  - Update project display/name references to "ml_playgound" (pyproject, docs, badges, READMEs).
  - Keep import package name stable for now to avoid breakage; perform code-level rename in Task 5.
- Acceptance criteria:
  - `pyproject.toml` project name updated; tooling and badges reflect the new name.
  - Docs and READMEs use the new project name consistently.
  - Quality gates pass.
- Commit guidance:
  - Branch: `chore/rename-project-ml_playgound`
  - Commit: `chore(project): rename project to ml_playgound in metadata and docs`

---

## Task 5: Switch to src/ layout and rename package folder

- Scope:
  - Move `ml_playground/` package under `src/ml_playgound/` (spelling per request) and update all tooling paths.
  - Update `pyproject.toml` to use `packages = [{include = "ml_playgound", from = "src"}]` (or equivalent for the chosen build backend).
  - Adjust import paths across the codebase and tests; fix path references in scripts and CI.
- Acceptance criteria:
  - Code imports from `ml_playgound` under `src/` successfully.
  - `make quality` passes; tests green; pyright/mypy updated to include `src/`.
  - No re-export shims left; `IMPORT_GUIDELINES.md` still satisfied.
- Commit guidance:
  - Branch: `refactor/src-layout-ml_playgound`
  - Commit: `refactor(pkg): move to src/ layout and rename package to ml_playgound`

---

## Task 6: Add a README for tools/

- Scope:
  - Create `tools/README.md` explaining purpose, structure, and how to run helper scripts (e.g., LIT setup, port_kill, gguf converter).
  - Document constraints (UV-only invocation) and example commands.
- Acceptance criteria:
  - `tools/README.md` exists with sections: Purpose, Structure, Usage, Examples.
  - References in `.dev-guidelines` and root `README.md` link to it where appropriate.
- Commit guidance:
  - Branch: `docs/tools-readme`
  - Commit: `docs(tools): add README for tooling and helper scripts`

## Archived import: previous Refactorings.md (snapshot 2025-10-01)

The following is an exact snapshot of `/.ldres/Refactorings.md` at migration time. Maintain new tasks above; this section is read-only.

```markdown
# Actionable Refactoring Todo List for nanoGPT

This document enumerates prioritized refactoring prompts that can be handed to code-generation agents. Each numbered task contains nested sub-goals to encourage small, reviewable commits.

---

## Agent-Ready, Granular Refactoring Prompts (Copy/Paste)

These prompts are self-contained and aligned with:

- `.dev-guidelines/DEVELOPMENT.md`
- `.dev-guidelines/IMPORT_GUIDELINES.md`
- `.dev-guidelines/GIT_VERSIONING.md`
- `.dev-guidelines/TESTING.md`
- `docs/framework_utilities.md`

Pre-commit hooks automatically run `make quality` during commits, so manual invocations are optional if you want faster feedback. Use short-lived feature branches, follow Conventional Commits, and always pair behavioral changes with tests. Never bypass verification (`--no-verify` is prohibited).

### Mandatory workflow for every prompt (no exceptions)

- **Create feature branch**: Update `main`, then create a kebab-case feature branch (`git switch -c <type>/<scope>-<desc>`).
- **Keep changesets tiny**: Implement the minimal behavior in ≤200 touched lines per commit, pairing tests and code.
- **Run quality gates**: Execute `make quality` (or stricter slices) locally before each commit and before opening the PR.
- **Open focused PR**: Push the branch, open a PR summarizing the change, list validation commands, and request review.
- **Merge cleanly into `main`**: After CI passes and review approval, use fast-forward or rebase-merge so the branch lands on `main` (a.k.a. master); delete the feature branch afterward.

---

## Index

- **P1** – Stage module boundary research for `ml_playground/model.py` ✅
- **P2** – Bootstrap modular package layout and migrate model imports ✅
- **P3** – Restructure test layout to mirror new module boundaries ✅
- **P4** – Segment trainer orchestration into dedicated subpackages ✅
- **P5** – Normalize data ingestion pipeline modules ✅
- **P6** – Align configuration and CLI layering ✅
- **P7** – Retire legacy entry points after package extraction ✅
- **P8** – Reconcile test tree with canonical package layout ✅
- **P9** – Update documentation to reflect canonical package structure 🔄
- **P10** – Consolidate cache directories under `.cache/` 🔄 (small; tool configs ✅)
- **P11** – Remove mocking via dependency injection in configuration classes 🔄 (medium)
- **P12** – Add README files to key subpackages ✅ (small)
- **P13** – Audit and fix misnamed/misplaced tests 🔄 (small)
- **P14** – Plan for `mutants/` directory management 🔄 (small; mutmut removed ✅)
- **P15** – Consolidate Python cache directories 🔄 (small, duplicate of P10; tool configs ✅)
- **P16** – Reorganize `ml_playground/` root utilities into subpackages 🔄 (large, do last)
- **P17** – Import compliance: remove re-exports and relative imports in `__init__.py` ✅ (high)
- **P18** – Consolidate LIT integration modules and docs ✅ (medium)
- **P19** – Harden checkpoint config serialization (exclude DI callables) 🔄 (small)

---

## Details

```

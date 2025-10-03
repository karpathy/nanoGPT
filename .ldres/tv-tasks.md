# tv-tasks

This document tracks Thomas's prioritized refactoring and infrastructure work. Keep entries concise,
reviewable, and compliant with our UV-first workflow (`make quality`). Reference
`.dev-guidelines/GIT_VERSIONING.md` for branch naming, Conventional Commits, and linear history, and
`.dev-guidelines/DOCUMENTATION.md` for documentation tone and formatting.

> **Identifier policy**: Only log tasks once a GitHub PR exists. Use the format `tv-YYYY-MM-DD:PR###`
> matching the PR number and creation date.

## Open tasks (updated 2025-10-03)

<!-- markdownlint-disable MD033 -->
<details>
<summary>Open task template (populate with existing PR)</summary>

```markdown
## Open · tv-YYYY-MM-DD:PR### · <Short title>

- **Summary**: Goal and scope in one or two sentences.
- **Priority**: P0/P1/P2/P3.
- **Size**: XS/S/M/L (estimate of work).
- **Meta?**: Yes/No — meta tasks unblock or shrink downstream work.
- **Dependencies**: Direct blockers or prerequisite tasks/PRs.
- **Next steps**: Ordered list of actionable sub-tasks.
- **Validation**: Commands required before commits/push.
- **Git plan**:
  - Branch: `<type>/<scope>-<desc>`
  - Commits: Conventional Commit messages with primary files/tests.
- **PR**:
  - Link: `<https://github.com/<org>/<repo>/pull/###>` (prefer Draft until ready).
  - Title suggestion
  - Body outline (Summary, Testing, Checklist)
```

</details>
<!-- markdownlint-enable MD033 -->

### Open · tv-2025-10-03:PR?? · Harmonize Python version requirements

- **Summary**: Align `pyproject.toml` and `.dev-guidelines/SETUP.md` Python version statements.
- **Priority**: P1
- **Size**: S
- **Meta?**: Yes — prevents environment drift.
- **Dependencies**: None; coordinate with tooling owners if version pin changes are needed.
- **Next steps**:
  1. Decide authoritative Python version target with stakeholders.
  2. Update `pyproject.toml`, documentation, and tooling configs accordingly.
  3. Run `make quality` and update CI matrices if required.
- **Validation**: `make quality`; targeted CI run if matrix changes.
- **Git plan**:
  - Branch: `chore/python-version-alignment`
  - Commits:
    - `chore(config): align python version requirements`
      (`pyproject.toml`, `.dev-guidelines/SETUP.md`, `.github/workflows/quality.yml` if needed)
- **PR**: Title `chore: align python version requirements`; body listing updated files and validation.

### Open · tv-2025-10-03:PR?? · Fix import guideline violations

- **Summary**: Resolve missing `from __future__ import annotations` declarations and relative imports.
- **Priority**: P0
- **Size**: M
- **Meta?**: Yes — ensures consistency for downstream work.
- **Dependencies**: Coordinate with module owners to avoid merge conflicts.
- **Next steps**:
  1. Add future annotations import to `ml_playground/model.py`, `ml_playground/datasets/__init__.py`, and
     `ml_playground/experiments/bundestag_qwen15b_lora_mps/__init__.py`.
  2. Replace relative imports in `ml_playground/tokenizer.py` and `ml_playground/analysis/__init__.py` with
     absolute paths per `IMPORT_GUIDELINES.md`.
  3. Run `make quality` and ensure `pyright` stays green.
- **Validation**: `make quality`.
- **Git plan**:
  - Branch: `fix/import-guidelines`
  - Commits:
    - `refactor(core): enforce import guidelines in tokenizer`
      (`ml_playground/tokenizer.py`, dependent call sites)
    - `refactor(analysis): replace relative import`
      (`ml_playground/analysis/__init__.py`)
    - `refactor(core): add future annotations import to model`
      (`ml_playground/model.py`)
    - `refactor(data): add future annotations import to datasets package`
      (`ml_playground/datasets/__init__.py`)
    - `refactor(experiments): add future annotations import to bundestag qwen package`
      (`ml_playground/experiments/bundestag_qwen15b_lora_mps/__init__.py`)
- **PR**: Title `refactor: enforce import guidelines`; body outlining files touched and validation.

### Open · tv-2025-10-03:PR?? · Establish regression test suite

- **Summary**: Consolidate regression tests under `tests/regression/`, add a README, and articulate scope
  to guard against AI-introduced regressions.
- **Priority**: P0
- **Size**: M
- **Meta?**: Yes — safeguards future changes.
- **Dependencies**: Documentation standards in `.dev-guidelines/DOCUMENTATION.md`; current regression tests
  spread across `tests/integration/` and `tests/unit/`.
- **Next steps**:
  1. Inventory existing regression-style tests; identify move/mirror targets.
  2. Create `tests/regression/README.md` following documentation guidelines and describe anti-regression
     policy.
  3. Relocate or re-export regression tests; adjust imports/fixtures as needed.
  4. Update `Makefile`/pytest configuration to include the new suite; ensure CI jobs reference it.
- **Validation**: `make quality`; `pytest tests/regression -q`.
- **Git plan**:
  - Branch: `test/regression-suite`
  - Commits:
    - `docs(tests): add regression suite readme` (`tests/regression/README.md`)
    - `test(regression): reorganize regression tests` (moved test files, `tests/__init__.py` updates)
    - `ci(makefile): add regression target` (`Makefile`, `.github/workflows/quality.yml`)
- **PR**: Title `test: establish regression suite`; body covering Summary, Testing, Checklist.

### Open · tv-2025-10-03:PR?? · Coverage roadmap towards ~100%

- **Summary**: Define realistic milestones and guardrails to reach ~100% coverage while accounting for
  stochastic tests.
- **Priority**: P1
- **Size**: M
- **Meta?**: Yes — strategy enables future coverage improvements.
- **Dependencies**: Stabilized badge workflow (deferred `tv-2025-10-03:PR35`); reproducible-build epic
  (`tv-2025-10-03:PR??`).
- **Next steps**:
  1. Document current coverage gaps with module-level breakdown.
  2. Classify gaps into deterministic vs stochastic segments; prioritize deterministic first.
  3. Draft incremental targets (e.g., 85%, 90%, 95%); codify acceptance criteria per milestone.
  4. Align CI dashboards to track milestones without flaking on stochastic components.
- **Validation**: `make coverage-test`; `make quality`.
- **Git plan**:
  - Branch: `docs/coverage-roadmap`
  - Commits:
    - `docs(coverage): outline roadmap to ~100 percent coverage`
      (`docs/coverage/roadmap.md`, `.ldres/tv-tasks.md` cross-reference)
- **PR**: Title `docs: define coverage roadmap`; body summarizing milestones, validation, and gating plan.

### Open · tv-2025-10-03:PR?? · Mutation testing initiative

- **Summary**: After coverage gating, introduce mutation-based testing (e.g., `cosmic-ray`) to measure test
  effectiveness.
- **Priority**: P1
- **Size**: M
- **Meta?**: Yes — improves future test quality.
- **Dependencies**: Coverage roadmap task; reproducible-build epic for tool integration.
- **Next steps**:
  1. Audit current mutation tool configs (remove `mutmut` residuals; align with `cosmic-ray`).
  2. Define initial mutation targets (critical modules) and thresholds for pass/fail.
  3. Integrate mutation runs into CI (nightly or gated) with summarized reporting.
- **Validation**: `make quality`; targeted `cosmic-ray` run.
- **Git plan**:
  - Branch: `test/mutation-suite`
  - Commits:
    - `chore(cosmic-ray): configure mutation testing targets` (`pyproject.toml`, `.github/workflows/mutation.yml`)
    - `docs(testing): document mutation workflow` (`docs/testing/mutation.md`)
- **PR**: Title `test: introduce mutation testing workflow`; body outlining configuration, validation, and
  follow-up tasks.

---

## Deferred tasks (updated 2025-10-03)

<!-- markdownlint-disable MD033 -->
<details>
<summary>Deferred task template (populate with existing PR)</summary>

```markdown
## Deferred · tv-YYYY-MM-DD:PR### · <Short title>

- **Summary**: Purpose and current status.
- **Priority**: P0/P1/P2/P3.
- **Size**: XS/S/M/L.
- **Meta?**: Yes/No — meta tasks unblock or shrink downstream work.
- **Dependencies**: Blockers or related initiatives.
- **References**: PRs, issues, or docs capturing detailed context.
- **Holding pattern**: Note what is awaited before resuming.
- **PR status**: Link and ensure it remains in Draft until ready to resume.
```

</details>

### Deferred · tv-2025-10-03:PR35 · Stabilize coverage badge workflow

- **Summary**: Badge generation differs between macOS and GitHub Actions because CI exercises an extra
  branch in `sample_batch()`. Investigation lives in PR #35.
- **Priority**: P0
- **Size**: M
- **Meta?**: No.
- **Dependencies**: Blocked on open task `tv-2025-10-03:PR??` (reproducible builds epic) before re-enabling the badge gate.
- **References**: Branch `coverage-badge-rebase`; PR #35 (in review).
- **Holding pattern**: Resume once the team approves the reproducibility plan captured in PR #35.
- **PR status**: `<https://github.com/<org>/<repo>/pull/35>` (keep in Draft until reproducible-build epic lands).
=======
- Scope:
  - In `Makefile`, replace indirections like `$(MAKE) pytest-core PYARGS="-m integration --no-cov"`
with direct invocations using `$(RUN) pytest` and `$(PYTEST_BASE)`.
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
  - Move or refactor these modules under their canonical package locations
(`ml_playground/training/checkpointing/`, `ml_playground/training/`, `ml_playground/models/utils/`
or `core/`, `ml_playground/training/optim/`, `ml_playground/core/`), aligned with prior restructurings.
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

## Notes

- `make quality` is the canonical gate and already runs pre-commit hooks
(ruff, format, checkmake, pyright, mypy, vulture,pytest).
No separate test run in the Make target is necessary.
- When the "remove mocks" effort is complete, enforce no mock usage via Ruff banned-modules and a CI grep sweep.

---

## Agent-Ready, Granular Refactoring Prompts (Copy/Paste)

These prompts are self-contained and aligned with:

- `.dev-guidelines/DEVELOPMENT.md`
- `.dev-guidelines/IMPORT_GUIDELINES.md`
- `.dev-guidelines/GIT_VERSIONING.md`
- `.dev-guidelines/TESTING.md`
- `docs/framework_utilities.md`

Pre-commit hooks automatically run `make quality` during commits,
so manual invocations are optional if you want faster feedback. Use short-lived feature branches,
follow Conventional Commits, and always pair behavioral changes with tests. Never bypass verification (`--no-verify` is prohibited).

### Mandatory workflow for every prompt (no exceptions)

- **Create feature branch**: Update `main`, then create a kebab-case feature branch (`git switch -c <type>/<scope>-<desc>`).
- **Keep changesets tiny**: Implement the minimal behavior in ≤200 touched lines per commit, pairing tests and code.
- **Run quality gates**: Execute `make quality` (or stricter slices) locally before each commit and before opening the PR.
- **Open focused PR**: Push the branch, open a PR summarizing the change, list validation commands, and request review.
- **Merge cleanly into `main`**: After CI passes and review approval,
use fast-forward or rebase-merge so the branch lands on `main` (a.k.a. master); delete the feature branch afterward.

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
  1. Python version target is 13.13.7.
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
      (`ml_playground/experiments/bundestag_qwen15b_lora_mps/__init__.py`)
- **PR**: Title `refactor: enforce import guidelines`; body outlining files touched and validation.

### Open · tv-2025-10-03:PR?? · Improve exception handling hygiene

- **Next steps**:
  1. Stage granular commits: checkpointing, training loop, and analysis LIT modules + task doc updates.
  2. Ensure `make quality` log captured for PR notes; gather relevant test artifacts if needed.
  3. Draft PR body summarizing exception-handling refinements, validation, and follow-up for tooling scripts.
  4. Update task tracker with PR number once available.
- **Validation**: `make quality`; relevant test targets.
- **Git plan**:
  - Branch: `refactor/error-handling`
  - Commits:
    - `refactor(data): tighten exception handling`
{{ ... }}
    - `refactor(prepare): tighten exception handling`
      (`ml_playground/prepare.py`)
    - `refactor(sampler): tighten exception handling`
      (`ml_playground/sampler.py`)
    - `refactor(trainer): tighten exception handling`
      (`ml_playground/trainer.py`)
    - `refactor(cli): tighten exception handling`
      (`ml_playground/cli.py`)
    - `refactor(config): tighten exception handling`
      (`ml_playground/config.py`)
- **PR**: Title `refactor: tighten exception handling`; body listing modules updated and tests run.
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

### Open · tv-2025-10-03:PR?? · Dev tooling quick reference

- **Summary**: Add a tooling section to `.dev-guidelines/DEVELOPMENT.md` covering `uvx`, `rg`, `gh`, `fzf`,
  and common non-interactive commands for workflow acceleration.
- **Priority**: P0
- **Size**: S
- **Meta?**: Yes — speeds up all future tasks.
- **Dependencies**: `.dev-guidelines/DEVELOPMENT.md`; existing command references.
- **Next steps**:
  1. Draft tooling overview emphasizing speed and reproducibility.
  2. Provide copy-pastable snippets for opening, closing, editing, running, rerunning, log retrieval, and
     status watching.
  3. Reference installation guidance; plan future automation for optional tooling installs.
- **Validation**: `make quality`.
- **Git plan**:
  - Branch: `docs/dev-tooling`
  - Commits:
    - `docs(development): add tooling quick reference`
      (`.dev-guidelines/DEVELOPMENT.md`)
- **PR**: Title `docs: add dev tooling quick reference`; body listing tools covered and validation.

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

- `make quality` is the canonical gate and already runs pre-commit hooks
(ruff, format, checkmake, pyright, mypy, vulture,pytest).
No separate test run in the Make target is necessary.
- When the "remove mocks" effort is complete, enforce no mock usage via Ruff banned-modules and a CI grep sweep.

---

## Task 8: Automate coverage badge generation ✅ (2025-10-02)

- Scope:
  - Add the `coverage-badge` CLI as a dependency and create a `Makefile` target `coverage-badge` that
  runs `uv run make coverage-report`, emits `coverage.xml`,
  and writes `docs/assets/coverage.svg`.
  - Add a local pre-commit hook that invokes the make target and stages the regenerated badge automatically.
  - Extend the GitHub Actions quality workflow to run the same target and fail if the committed badge diverges.
  - Update `README.md` to render the badge and document the workflow in `.dev-guidelines/TESTING.md`.
- Acceptance criteria:
  - Every commit (local and CI) regenerates and stages an up-to-date `docs/assets/coverage.svg`.
  - `make coverage-badge` is idempotent and shared across local and CI workflows.
  - README badge always reflects latest coverage; quality gate ensures badge freshness.
- Commit guidance:
  - Branch: `chore/coverage-badge`
  - Commit 1: `chore(makefile): add coverage badge target`
  - Commit 2: `chore(ci): enforce coverage badge freshness`
  - Commit 3: `docs(readme): add coverage badge`

---

## Coverage status (2025-10-02)

- **Local coverage snapshot**: `ml_playground/training/checkpointing/service.py` remains at 66.27% with
  missed lines `[76-82, 109-110, 117-119, 138-141, 144, 154-156]` per `.cache/coverage/coverage.json`.
- **CI artifact**: GitHub runner covers the same file at 93.98% (only line `144` missed) according to
- **Current decision**: Pause further coverage/badge work until the test suite complies with the
  mock-free policy; revisit after completing Tasks 9–13 and rebase coverage changes on top of the
  updated fixtures.

---

## Task 9: Replace monkeypatches in training loop unit tests (2025-10-03)

- Scope:
  - Refactor `tests/unit/training/loop/test_training_runner.py` to comply with `.dev-guidelines/TESTING.md` and `tests/unit/README.md`
  guidance against monkeypatching internal seams.
    `tests/unit/conftest.py` and inject collaborators via constructor or
    dependency overrides instead of `monkeypatch.setattr`.
  - Ensure `Trainer` can accept injected dependencies without modifying production behavior for
    non-test callers (consider helper factory or fixture that wires dependencies).
  - Remove temporary allowlist entry for `tests/unit/core/test_checkpoint.py` once mocks are gone.
- Acceptance criteria:
  - The test file contains no `monkeypatch` usage or direct attribute patching of `runner_mod`.
{{ ... }}
  - `make unit` and `make quality` remain green.
- Commit guidance:
  - Branch: `test/training-loop-fixture-refactor`
  - Commit: `test(training): replace monkeypatches with fixtures in training_runner tests`

Status: Completed.
`tests/unit/training/loop/test_training_runner.py` already uses dependency injection
via `TrainerDependencies` and contains no `monkeypatch` usage.

---

## Task 10: Remove monkeypatch usage from configuration CLI tests ✅ (2025-10-03)

- Scope:
  - Update `tests/unit/configuration/test_cli.py` to stop patching CLI internals and environment via
    `monkeypatch`/`MagicMock`.
  - Provide fixtures for temporary configs, logging sinks, and checkpoint directories, leveraging
    helper utilities instead of runtime patching.
- Acceptance criteria:
  - No `monkeypatch`, `patch`, or `MagicMock` references remain in the file; collaborators come from fixtures
  or lightweight fakes defined alongside the tests.
  - CLI tests continue to verify error handling and success paths without altering global state.
  - `make unit` stays under target runtime.
- Commit guidance:
  - Branch: `test/config-cli-fixtures`
  - Commit: `test(configuration): replace monkeypatch usage with fixtures in CLI tests`

Status: Completed.
Refactored `tests/unit/configuration/test_cli.py` to use DI seams. Introduced
`default_config_path` parameter in loaders and `cuda_is_available` injection for
`cli._global_device_setup()`; no `monkeypatch` remains.

---

## Task 11: Refactor sampling runner unit tests to fixture-based fakes ✅ (2025-10-03)

- Scope:
  - Rework `tests/unit/sampling/test_runner.py` to remove `monkeypatch` usage for tokenizer setup,
    checkpoint loading, and sampler orchestration.
  - Move shared doubles into fixtures (e.g., fake tokenizer registry, fake checkpoint files) stored
    in `tests/unit/sampling/conftest.py`.
- Acceptance criteria:
  - The file no longer references `monkeypatch` or related patch APIs.
  - Tests still validate sequential/random sampling flows and error handling using fixture-provided collaborators.
  - Coverage is unchanged or improved; `make unit` passes.
- Commit guidance:
  - Branch: `test/sampling-runner-fixtures`
  - Commit: `test(sampling): eliminate monkeypatch in runner tests`

Status: Completed.
`tests/unit/sampling/test_runner.py` now uses dependency injection via
`SamplerConfig.checkpoint_load_fn`, `SamplerConfig.model_factory`, and
`SamplerConfig.compile_model_fn`. All direct monkeypatching of `GPT`,
`torch.compile`, and internal methods has been removed.

---

## Task 12: Convert experiments loader tests away from monkeypatch ✅ (2025-10-03)

- Scope:
  - Modify `tests/unit/experiments/test_experiments_loader.py` to replace `monkeypatch.setattr` with
    fixture-driven module registries and temporary package structures.
  - Use `tmp_path` to create synthetic experiment modules/files and load them through the public API.
- Acceptance criteria:
  - No runtime attribute patching remains; fixtures encapsulate filesystem setup and module registration.
  - Tests continue to cover success and failure branches for experiment loading.
  - `make unit` succeeds without performance regressions.
- Commit guidance:
  - Branch: `test/experiments-loader-fixtures`
  - Commit: `test(experiments): replace monkeypatch with fixtures`

Status: Completed.
Added DI parameters to `ml_playground/experiments/registry.load_preparers()`
for `resources` and `import_module`; updated
`tests/unit/experiments/test_experiments_loader.py` to use DI and removed
`monkeypatch` usage.

---

## Task 13: Align integration datasets tests with mock-free policy

- Scope:
  - Audit `tests/integration/test_datasets_shakespeare.py` and remove `monkeypatch`/`patch` usage,
    replacing them with fixtures that supply in-memory datasets, clocks, and file handles.
  - Ensure integration tests mimic real data flow using temporary directories and deterministic sample
    data per `.dev-guidelines/TESTING.md`.
- Acceptance criteria:
  - Integration tests run without patching internal functions; all collaborators come from fixtures or helper modules.
  - Tests remain under the 100ms performance target and continue to validate download/encode behavior.
  - `make integration` passes.
- Commit guidance:
  - Branch: `test/integration-shakespeare-fixtures`
  - Commit: `test(integration): remove monkeypatch from shakespeare dataset tests`

---

This document enumerates prioritized refactoring prompts that can be handed to code-generation agents.
Each numbered task contains nested sub-goals to encourage small, reviewable commits.

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

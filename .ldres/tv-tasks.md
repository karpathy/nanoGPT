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

- **Summary**: Chart a steady path toward 100% coverage with deterministic, guideline-compliant
  tests. Roadmap lives in `docs/coverage/roadmap.md`.
- **Size**: M
- **Meta?**: Yes — strategy enables future coverage improvements.
- **Dependencies**: Stabilized badge workflow (deferred `tv-2025-10-03:PR35`); reproducible-build epic
  (`tv-2025-10-03:PR??`).
- **Next steps**:
  1. Generate a per-module coverage scoreboard and file tasks for gaps identified.
  2. Work through deterministic modules first (CLI/config/data pipeline) using DI and fakes.
  3. Extend coverage to experiments and runtime surfaces while honoring runtime budgets.
  4. Gradually raise coverage gates (95% → 99% → 100%) in sync with completed milestones, logging each bump.
  5. Migrate eligible deterministic unit tests to property-based suites with clear oracles.
  6. Refactor property-based tests to reuse fixtures and shrink shared setup code.
- **Latest snapshot (2025-10-05)**:
  - Global coverage 79.06% (`make coverage-report` running unit + property suites).
  - Pre-commit gate set to `--fail-under=79.00`; raise ≥81.50% after the next Milestone 1 uplift.
- **Module backlog** (open coverage tasks):
  - `ml_playground/data_pipeline/preparer.py`
  - `ml_playground/cli.py`
  - `ml_playground/configuration/loading.py`
  - `ml_playground/models/core/inference.py`
  - `ml_playground/training/ema.py`
  - `ml_playground/experiments/bundestag_qwen15b_lora_mps/preparer.py`
  - `ml_playground/experiments/bundestag_tiktoken/preparer.py`
  - `ml_playground/experiments/bundestag_char/preparer.py`
  - `ml_playground/experiments/speakger/preparer.py`
  - `ml_playground/experiments/speakger/sampler.py`
  - `ml_playground/training/hooks/logging.py`
  - `ml_playground/training/hooks/runtime.py`
  - `ml_playground/core/tokenizer_protocol.py`
  - `ml_playground/core/logging_protocol.py`
- **Git plan**:
  - Branch: `docs/coverage-roadmap`
  - Commits:
    - `docs(coverage): outline roadmap to ~100 percent coverage`
      (`docs/coverage/roadmap.md`, `.ldres/tv-tasks.md` cross-reference)
- **PR**: Title `docs: define coverage roadmap`; body summarizing milestones, validation, and gating plan.

### Open · tv-2025-10-05:PR?? · Markdownlint tooling harmonization

- **Summary**: Align markdownlint execution with the pyproject.toml-only configuration policy. Replace the
  current `markdownlint-cli2` hook with a solution that respects centralized configuration.
- **Priority**: P1
- **Size**: S
- **Meta?**: Yes — removes a recurring exception to the tooling policy.
- **Dependencies**: None.
- **Next steps**:
  1. Research markdown lint tools that read settings from `pyproject.toml` (or can be wrapped to do so).
  2. Prototype the replacement hook locally and document migration steps.
  3. Update `.githooks/.pre-commit-config.yaml` and supporting docs; ensure CI parity.
- **Validation**: `make docs-lint` (or equivalent new target) once introduced.
- **Git plan**:
  - Branch: `build/markdownlint-pyproject`
  - Commit: `build(markdownlint): adopt pyproject-based lint runner`
  - **Status**: Pending research
- **PR**: Title `build: harmonize markdownlint with pyproject tooling`; include comparison runs and rollout notes.

### In flight · tv-2025-10-05:fixtures · Property suite fixture consolidation

- **Summary**: Move shared helpers (config factories, attribute overrides) into
  `tests/conftest.py` and refactor property/unit suites to consume them.
- **Priority**: P1
- **Size**: S
- **Meta?**: No
- **Dependencies**: None.
- **Next steps**:
  1. Apply `shared_config_factory` and `override_attr` to remaining property suites (configuration, data pipeline, etc.).
  2. Extract any repetitive Hypothesis strategies into reusable helpers.
  3. Follow up with unit suites once property suites converge on shared fixtures.
- **Validation**: `uv run pytest tests/property` and targeted unit suites after refactor.
- **Git plan**:
  - Branch: `test/fixtures-unification`
  - Commit: `test(fixtures): centralize property helpers`
  - **Status**: Shared fixtures added; CLI property suite migrated.
- **PR**: Title `test: consolidate shared fixtures across property suites`; include before/after duplication metrics.

### In progress · tv-2025-10-05:mutation · Introduce mutation testing

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

### Deferred · tv-2025-10-04:PR?? · Expand CI Python version matrix

- **Summary**: Add optional GitHub Actions matrix to exercise supported Python versions beyond 3.13 once
  runtime stability and coverage reach 100%.
- **Priority**: P2
- **Size**: S
- **Meta?**: Yes — improves long-term compatibility validation.
- **Dependencies**: Requires CI runtime optimizations (uv caching, artifact reuse) and stabilized coverage thresholds.
- **References**: `.github/workflows/quality.yml` (current single-version job).
- **Holding pattern**: Defer until coverage roadmap milestones complete and multi-version support becomes necessary.
- **PR status**: Not yet created; future branch could be `ci/python-matrix`.

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

# tv-tasks

This document tracks Thomas's prioritized refactoring and infrastructure work. Keep entries concise,
reviewable, and compliant with our UV-first workflow (`uv run ci-tasks quality`). Reference
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
  1. Create `tests/regression/README.md` following documentation guidelines and describe anti-regression
     policy.
  1. Relocate or re-export regression tests; adjust imports/fixtures as needed.
  1. Update `test-tasks`/pytest configuration to include the new suite; ensure CI jobs reference it.
- **Validation**: `uv run ci-tasks quality`; `uv run pytest tests/regression -q`.
- **Git plan**:
  - Branch: `test/regression-suite`
  - Commits:
    - `docs(tests): add regression suite readme` (`tests/regression/README.md`)
    - `test(regression): reorganize regression tests` (moved test files, `tests/__init__.py` updates)
    - `ci(ci-tasks): add regression target` (`tools/ci_tasks.py`, `.github/workflows/quality.yml`)
- **PR**: Title `test: establish regression suite`; body covering Summary, Testing, Checklist.

### Open · tv-2025-10-03:PR?? · Coverage roadmap towards ~100%

- **Summary**: Deliver a lean roadmap for closing remaining coverage gaps and ratcheting gates to 100%. Source of truth: `docs/coverage/roadmap.md`.
- **Size**: M
- **Meta?**: Yes — enables future coverage improvements.
- **Dependencies**: Stabilized badge workflow (deferred `tv-2025-10-03:PR35`); reproducible-build epic (`tv-2025-10-03:PR??`).
- **Next steps**:
  1. Finish deterministic gaps:
     - `ml_playground/cli.py` (86.97%): cover CLI dispatch and error paths.
     - `ml_playground/data_pipeline/transforms/tokenization.py` (92.96%): exercise exception handlers.
     - `ml_playground/training/checkpointing/{service.py, checkpoint_manager.py}` (~94%): test defensive branches.
  1. Address protocol and hardware bottlenecks:
     - `ml_playground/core/tokenizer_protocol.py` and `core/logging_protocol.py`: add interface contract tests via implementations.
     - `ml_playground/training/hooks/runtime.py`: isolate GPU-only logic and document skip strategy.
  1. Record milestones in `docs/coverage/roadmap.md`, raising `--fail-under` with each completed cluster (90% → 95% → 99% → 100%).
  1. Backfill roadmap issues for any residual \<100% modules before the final gate raise.
- **Latest snapshot (2025-10-05)**:
  - Global coverage **87.28%** (`uv run ci-tasks coverage-report --fail-under 87.00`).
  - Pre-commit gate `--fail-under=87.00`.
  - `.ldres/coverage-opportunities.md` holds module-level notes.
- **Git plan**:
  - Commits:
    - `docs(coverage): update roadmap with remaining coverage milestones`
- **PR**: Title `docs: refine coverage roadmap`; body summarizing remaining deltas and gating plan.

### Open · tv-2025-10-05:PR?? · Accelerate test execution

- **Summary**: Profile the performance of every automated test suite and implement improvements so local and CI feedback loops stay fast.
- **Priority**: P1
- **Size**: M
- **Meta?**: Yes — speeds up developer feedback.
- **Dependencies**: None.
- **Latest baseline (2025-10-05)**:
  - `uv run pytest tests/unit -n 0 --maxfail=1 --durations=20` → 2.26s
    - Slowest cases: `tests/unit/data_pipeline/test_memmap.py::TestMemmapReader::test_memmap_reader_creation` (~0.13s) and related memmap sampling tests (~0.05–0.11s). `test_initialize_components_with_compile` is now <0.01s and CLI exit-path tests remain sub-0.01s.
  - `uv run pytest tests/property -n 0 --maxfail=1 --durations=10` → 4.34s
    - Slowest cases: `TestMergeMappings` methods (0.25–0.90s) after trimming Hypothesis workloads; `mdformat` adjustments keep docs stable.
  - `uv run pytest tests/integration -n 0 --maxfail=1 --durations=20` → 0.08s
    - No standout slowcases; Shakespeare dataset setup tops at 0.02s.
  - `uv run pytest tests/acceptance -n 0 --maxfail=1 --durations=20` → 0.93s
    - `tests/acceptance/steps/test_checkpointing_steps.py::test_keep_policy_enforcement_for_last_checkpoints` ~0.05s remains the peak.
  - `uv run pytest tests/e2e -n 0 --maxfail=1 --durations=10` → 2.88s
    - `test_train_bundestag_char_quick` dominates at 0.10s; configs stay minimal and documented.
  - `make quality` (pre-commit bundle) → 23.39s wall-clock (clean tree).
  - `make quality-fast` (format-only hooks) → 1.38s wall-clock after format passes clean.
- **Next steps**:
  1. Investigate memmap-heavy tests (`tests/unit/data_pipeline/test_memmap.py`, `tests/unit/data_pipeline/test_sampling.py`) for fixture reuse or lighter data to shave remaining 0.1–0.2s hotspots.
  1. Explore smaller `max_examples` or strategy caching for `TestMergeMappings` if further property-suite savings are required; ensure assertions still meaningful.
  1. Track additional optimizations (e.g., memmap fixture changes) and rerun `pytest --durations=20` after each iteration; update this log with new baselines.
- **Validation**: `uv run ci-tasks quality`; targeted `pytest --durations=20` reruns on affected suites.
- **Git plan**:
  - Commits:
    - `ci(pytest): profile test runtimes and document hotspots`
    - `ci(pytest): apply performance optimizations`
- **PR**: Title `ci: accelerate test execution`; body covering baseline, optimizations, and measured wins.

- **Dependencies**: Coverage roadmap task; reproducible-build epic for tool integration.
- **Latest baseline (2025-10-06)**:
  - `uv run ci-tasks mutation run` (module-path `ml_playground/`, timeout 1 s) processed **5 314** mutants (killed: 5 312,
    incompetent: 2) in ~**1 h 31 m** wall clock.
- **Next steps**:
  1. Ship an automation pipeline that ingests `cosmic-ray` survivors and opens draft PRs with proposed fixes
     after long-running CI jobs (script + workflow).
  1. Triage surviving mutants by module and harden tests iteratively:
     - `ml_playground/training/checkpointing/`
     - `ml_playground/core/`
     - `ml_playground/models/`
     - `ml_playground/data_pipeline/`
     - `ml_playground/sampling/`
  1. Track survivor counts per module after each run and update this task with progress metrics.
- **Validation**: `uv run ci-tasks mutation run`; targeted survivor-specific pytest slices.
- **Git plan**:
  - Branch: `test/mutation-hardening`
  - Commits:
    - `chore(cosmic-ray): automate survivor extraction` (`tools/`, `.github/workflows/`)
    - `test(<module>): strengthen tests against survivor diff`
    - `docs(testing): update mutation workflow guidance`
- **PR**: Title `test: harden mutation workflow`; body covering automation, survivor stats, and validation.

### Open · tv-2025-10-06:PR?? · Align READMEs with cross-doc details policy

- **Summary**: Apply the new `Related documentation` details-block template across remaining README files so cross-links stay consistent and summarized.
- **Priority**: P2
- **Size**: S
- **Meta?**: Yes — enforces documentation guidelines.
- **Dependencies**: New policy in `.dev-guidelines/DOCUMENTATION.md`.
- **Next steps**:
  1. Audit README files beyond `tests/unit/` and `tests/property/` for related-doc references.
  1. Insert or update `<details>` blocks with relative links and first-paragraph summaries per the template.
  1. Note any documents lacking clear intro paragraphs and schedule follow-up edits if needed.
- **Validation**: `uv run ci-tasks quality` (mdformat + lint).
- **Git plan**:
  - Branch: `docs/related-details-rollout`
  - Commits:
    - `docs(guidelines): apply related-document template across READMEs`
- **PR**: Title `docs: align READMEs with related documentation template`; body summarizing updated files and guideline compliance.

### Open · tv-2025-10-11:PR?? · Replace mocks with DI-first tests

- **Summary**: Eliminate remaining mocking across test suites by introducing DI-friendly fakes/fixtures and shrinking the `.no-mock-allowlist` to zero.
- **Priority**: P1
- **Size**: L
- **Meta?**: Yes — enforces long-term testing discipline.
- **Dependencies**: `.dev-guidelines/TESTING.md`; `.no-mock-allowlist`; refactoring bandwidth from active PRs.
- **Next steps**:
  1. Audit `.no-mock-allowlist` and catalog mocking patterns per file.
  1. Replace `config_loading` monkeypatching in `tests/unit/configuration/test_models_and_loading.py` with fixture-driven path injection.
  1. Refactor training and checkpointing tests to use DI fakes instead of `monkeypatch`.
  1. Update analysis/LIT tests with deterministic fakes; delete each allowlist entry as refactors land.
  1. Document the anti-mock playbook in `.dev-guidelines/TESTING.md` and broadcast via PR notes.
- **Validation**: `uv run env-tasks verify`; targeted `uv run pytest tests/unit -n 0`.
- **Git plan**:
  - Branch: `test/remove-mocks`
  - Commits:
    - `test(<area>): replace mocks with DI fakes`
    - `docs(testing): codify DI-first testing playbook`
- **PR**: Title `test: replace mocks with DI-first fixtures`; body covering scope, updated allowlist, and validation.

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

______________________________________________________________________

## Notes

- `uv run ci-tasks quality` is the canonical gate and already runs pre-commit hooks
  (ruff, format, checkmake, pyright, mypy, vulture,pytest).
  No separate test run in the Make target is necessary.
- When the "remove mocks" effort is complete, enforce no mock usage via Ruff banned-modules and a CI grep sweep.

______________________________________________________________________

## Agent-Ready, Granular Refactoring Prompts (Copy/Paste)

These prompts are self-contained and aligned with:

- `.dev-guidelines/DEVELOPMENT.md`
- `.dev-guidelines/IMPORT_GUIDELINES.md`
- `.dev-guidelines/GIT_VERSIONING.md`
- `.dev-guidelines/TESTING.md`
- `docs/framework_utilities.md`

Pre-commit hooks automatically run `uv run ci-tasks quality` during commits,
so manual invocations are optional if you want faster feedback. Use short-lived feature branches,
follow Conventional Commits, and always pair behavioral changes with tests. Never bypass verification (`--no-verify` is prohibited).

### Mandatory workflow for every prompt (no exceptions)

- **Create feature branch**: Update `main`, then create a kebab-case feature branch (`git switch -c <type>/<scope>-<desc>`).
- **Keep changesets tiny**: Implement the minimal behavior in ≤200 touched lines per commit, pairing tests and code.
- **Run quality gates**: Execute `uv run ci-tasks quality` (or stricter slices) locally before each commit and before opening the PR.
- **Open focused PR**: Push the branch, open a PR summarizing the change, list validation commands, and request review.
- **Merge cleanly into `main`**: After CI passes and review approval,
  use fast-forward or rebase-merge so the branch lands on `main` (a.k.a. master); delete the feature branch afterward.

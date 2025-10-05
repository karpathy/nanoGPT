# Coverage Roadmap

Last updated: 2025-10-05

## Baseline snapshot

- Overall line coverage (2025-10-05): **87.28%**\
  (`make coverage-report` on Python 3.13.5; unit + property suites)
- Pre-commit gate: `coverage report --fail-under=87.00`
  (raise alongside completed milestones: 90 → 95 → 99 → 100)
- Coverage artifacts stored under `.cache/coverage/coverage.sqlite`

## Policy alignment

- `.dev-guidelines/TESTING.md` targets **100% line and branch coverage per module** with deterministic, mock-free tests.
- Transitional gates are acceptable while concrete gaps are being closed, but every milestone should tighten enforcement.
- Tests must remain within documented runtime budgets and avoid test-only production branches.

## Remaining gaps

- **Deterministic modules**

  - `ml_playground/cli.py` (86.97%): cover CLI dispatch, option validation, and error messaging.
  - `ml_playground/data_pipeline/transforms/tokenization.py` (92.96%): exercise exception and retry handlers.
  - `ml_playground/training/checkpointing/service.py` (94.12%) & `checkpoint_manager.py` (93.17%): add defensive-path assertions.

- **Protocols and shared interfaces**

  - `ml_playground/core/tokenizer_protocol.py` (75.00%): expand contract tests through concrete implementations.
  - `ml_playground/core/logging_protocol.py` (76.47%): validate adapters via fake sinks.

- **Hardware-sensitive surfaces**

  - `ml_playground/training/hooks/runtime.py` (78.57%): isolate GPU-specific logic and document skip strategy.

`.ldres/coverage-opportunities.md` tracks deeper notes and owners for any follow-up sub-tasks.

## Milestones

1. **Finish deterministic gaps**

   - Land additional unit/property coverage for CLI, tokenization transforms, and checkpoint services.
   - Capture before/after metrics and raise `--fail-under` to 90 once merged.

1. **Exercise protocol interfaces**

   - Write contract-focused tests for tokenizer/logging protocols via their concrete implementations.
   - Document patterns for future protocol additions.
   - Raise `--fail-under` to 95 upon completion.

1. **Tame hardware-dependent code**

   - Provide CPU-backed or stubbed tests for `training/hooks/runtime.py`, including GPU skip rationale.
   - Record gating guidance for CI environments lacking accelerators.
   - Increase coverage gate to 99%, leaving only residual edge cases.

1. **Ratcheting to 100%**

   - Open issues for any remaining \<100% modules, then close them out.
   - Once every tracked module reports 100% line/branch coverage, set all gates (pre-commit, CI, badges) to 100% and keep them locked.
   - Maintain historical coverage updates in `.ldres/tv-tasks.md` and announce milestones during stand-ups.

## Threshold management

- Track gate adjustments in `.githooks/.pre-commit-config.yaml` and coordinate PR messaging.
- Mirror threshold increases in CI workflows and update contributor docs to reflect new expectations.
- Use `coverage json` outputs to spot regressions whenever thresholds are raised.

## Tracking & ownership

- DRI: Thomas (`tv`).
- Status updates recorded in `.ldres/tv-tasks.md` under `tv-2025-10-03:PR?? · Coverage roadmap towards ~100%`.
- Related initiatives: Regression suite (tv-2025-10-03:PR??), Mutation testing (tv-2025-10-05:mutation), Accelerate test execution (tv-2025-10-05:PR??).

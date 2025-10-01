# Coverage Baseline

_Last updated: 2025-10-01_

This document captures the current automated test coverage for the `ml_playground` package and describes how to regenerate the reports. Use it to prioritize work that pushes statement and branch coverage towards 100%.

---

## Generating the baseline

- **Prerequisites**
  - Run inside the project root with the UV-managed virtual environment available.
- **Command**

  ```bash
  make coverage-report
  ```

- **Artifacts**
  - Terminal summary emitted during the run.
  - HTML report at `.cache/coverage/htmlcov/index.html` (open in a browser).
  - JSON report at `.cache/coverage/coverage.json` (machine-readable; used for scripting).
- **Notes**
  - The Make target runs `pytest` serially (`-n 0`) with the `not perf` marker filter.
  - Coverage reports temporarily override `fail_under` to zero so artifacts always generate, even while total coverage is below the ultimate goal. CI guardrails will be restored once coverage approaches 100%.

---

## Lowest-covered modules (statement %)

The table below lists modules below 90% statement coverage, extracted from `.cache/coverage/coverage.json` after running `make coverage-report` on 2025-10-01.

| Module | Statements | Missing | Coverage |
| --- | ---: | ---: | ---: |
| `ml_playground/training/ema.py` | 16 | 8 | 40.00% |
| `ml_playground/training/checkpointing/service.py` | 63 | 27 | 54.22% |
| `ml_playground/data_pipeline/preparer.py` | 83 | 49 | 33.66% |
| `ml_playground/models/core/inference.py` | 43 | 18 | 56.14% |
| `ml_playground/training/loop/runner.py` | 136 | 25 | 78.92% |
| `ml_playground/core/logging_protocol.py` | 9 | 0 (branches) | 76.47% |
| `ml_playground/core/tokenizer_protocol.py` | 14 | 2 | 75.00% |
| `ml_playground/sampling/runner.py` | 98 | 17 | 78.69% |
| `ml_playground/training/hooks/logging.py` | 11 | 2 | 76.92% |
| `ml_playground/experiments/bundestag_qwen15b_lora_mps/preparer.py` | 44 | 35 | 15.52% |
| `ml_playground/experiments/bundestag_qwen15b_lora_mps/sampler.py` | 10 | 10 | 0.00% |
| `ml_playground/experiments/bundestag_qwen15b_lora_mps/trainer.py` | 10 | 10 | 0.00% |
| `ml_playground/experiments/bundestag_tiktoken/preparer.py` | 40 | 29 | 27.50% |
| `ml_playground/experiments/speakger/preparer.py` | 13 | 6 | 53.85% |
| `ml_playground/experiments/speakger/sampler.py` | 99 | 21 | 78.99% |

_Additional modules may fall below 90% once new code lands; regenerate the report before starting each coverage sprint._

---

## Using the baseline in PRs

- **Before coding**: Review the table (and HTML report) to choose a module with the largest gap.
- **During development**: Run targeted tests frequently, then `make coverage-report` before opening the PR to ensure deltas are captured.
- **In PR description**: Mention the new coverage percentage for the touched module(s) and link to relevant test files.
- **Post-merge**: Re-run `make coverage-report` on `master` to keep the baseline current.
- **Fixture discipline**: Follow the guidance in `.dev-guidelines/TESTING.md#fixtures-strict-usage` and `tests/unit/README.md#fixtures--collaborators`; prefer inline stubs/DI over monkeypatching or new fixtures unless they belong in an existing `conftest.py`.

---

## Next steps

- **Task 2** (`/.ldres/tv-tasks.md`): Raise unit coverage for `ml_playground/training/checkpointing/service.py`, `ml_playground/training/loop/runner.py`, and `ml_playground/core/tokenizer.py` to ≥99% statements / ≥95% branches.
- **Task 3**: Strengthen CLI and E2E coverage for `ml_playground/cli.py` and related entry points.
- **Task 4**: Once coverage gaps shrink, re-enable strict `fail_under` thresholds in CI.

# Coverage Baseline

Last updated: 2025-10-02 (after DI trainer refactor)

This document captures the current automated test coverage for the `ml_playground` package and describes how to
regenerate the reports. Use it to prioritize work that pushes statement and branch coverage towards 100%.

---

## Quick workflow recap

1. Run `make coverage-report` from the project root (UV-managed environment active).
2. Inspect the terminal summary and open `.cache/coverage/htmlcov/index.html` for interactive drill-down.
3. Extract module-level metrics from `.cache/coverage/coverage.json` when updating this baseline.

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
  - Coverage reports temporarily override `fail_under` to zero so artifacts always generate, even while total coverage
    is below the ultimate goal. CI guardrails will be restored once coverage approaches 100%.

---

## Lowest-covered modules (statement %)

The modules below fell under 90% statement coverage when `make coverage-report` was run on 2025-10-02 (data from
`.cache/coverage/coverage.json`):

- **`ml_playground/training/ema.py`** — statements: 16, missing: 8, coverage: 40.00%.
- **`ml_playground/training/checkpointing/service.py`** — statements: 63, missing: 20, coverage: 66.27%.
- **`ml_playground/data_pipeline/preparer.py`** — statements: 83, missing: 49, coverage: 33.66%.
- **`ml_playground/models/core/inference.py`** — statements: 43, missing: 18, coverage: 56.14%.
- **`ml_playground/training/loop/runner.py`** — statements: 176, missing: 32, coverage: 87.02%.
- **`ml_playground/core/logging_protocol.py`** — statements: 9, branch misses only, coverage: 76.47%.
- **`ml_playground/core/tokenizer_protocol.py`** — statements: 14, missing: 2, coverage: 75.00%.
- **`ml_playground/sampling/runner.py`** — statements: 98, missing: 17, coverage: 78.69%.
- **`ml_playground/training/hooks/logging.py`** — statements: 11, missing: 2, coverage: 76.92%.
- **Bundestag Qwen 1.5B LoRA preparer** (`experiments/bundestag_qwen15b_lora_mps/preparer.py`) — statements: 44,
  missing: 35, coverage: 15.52%.
- **Bundestag Qwen 1.5B LoRA sampler** (`experiments/bundestag_qwen15b_lora_mps/sampler.py`) — statements: 10,
  missing: 10, coverage: 0.00%.
- **Bundestag Qwen 1.5B LoRA trainer** (`experiments/bundestag_qwen15b_lora_mps/trainer.py`) — statements: 10,
  missing: 10, coverage: 0.00%.
- **Bundestag Tiktoken preparer** (`experiments/bundestag_tiktoken/preparer.py`) — statements: 40, missing: 29,
  coverage: 27.50%.
- **SpeakGER preparer** (`experiments/speakger/preparer.py`) — statements: 13, missing: 6, coverage: 53.85%.
- **SpeakGER sampler** (`experiments/speakger/sampler.py`) — statements: 99, missing: 21, coverage: 78.99%.

_Additional modules may fall below 90% once new code lands; regenerate the report before starting each coverage sprint._

---

## Using the baseline in PRs

- **Before coding**: Review the table (and HTML report) to choose a module with the largest gap.
- **During development**: Run targeted tests frequently, then `make coverage-report` before opening the PR to ensure
  deltas are captured.
- **In PR description**: Mention the new coverage percentage for the touched module(s) and link to relevant test files.
- **Post-merge**: Re-run `make coverage-report` on `master` to keep the baseline current.
- **Fixture discipline**: Follow the guidance in
  `.dev-guidelines/TESTING.md#fixtures-strict-usage` and
  `tests/unit/README.md#fixtures--collaborators`; prefer inline stubs/DI over monkeypatching or new fixtures unless
  they belong in an existing `conftest.py`.

---

## Next steps

- **Task 2** (`/.ldres/tv-tasks.md`): Raise unit coverage for
  `ml_playground/training/checkpointing/service.py`, `ml_playground/training/loop/runner.py`, and
  `ml_playground/core/tokenizer.py` to ≥99% statements / ≥95% branches.
- **Task 3**: Strengthen CLI and E2E coverage for `ml_playground/cli.py` and related entry points.
- **Task 4**: Once coverage gaps shrink, re-enable strict `fail_under` thresholds in CI.

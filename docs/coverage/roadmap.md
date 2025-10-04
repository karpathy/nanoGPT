# Coverage Roadmap

Last updated: 2025-10-04

## Baseline snapshot

- Overall line coverage (2025-10-04): **82.35%**
  (`make coverage-report` under Python 3.13.5)
- Pre-commit gate: `coverage report --fail-under=81.50`
  (CI variance buffer)
- Coverage data stored under `.cache/coverage/coverage.sqlite`

## Policy alignment

- `.dev-guidelines/TESTING.md` mandates **100% line and branch coverage per module** with
  no exceptions or test-specific production code paths.
- Transitional gating remains acceptable while deficits are being retired, but the
  long-term goal is full compliance. Coverage rises should happen steadily without
  compromising determinism or quality.
- Testing must remain deterministic, mock-free, and within the runtime budgets defined
  in the guidelines.

## Gap analysis

- **High-impact deterministic gaps**
  - `ml_playground/cli.py` (73.62%): CLI option error paths, dataset downloads,
    and project scaffolding flows lack unit coverage.
  - `ml_playground/configuration/loading.py` (80.00%): Config file fallbacks and
    environment-variable overrides currently rely on ad-hoc experimentation.
  - `ml_playground/data_pipeline/preparer.py` (33.66%): File IO and metadata
    generation branches are untested; cover with temp directories and fake
    tokenizer fixtures.
  - Experiment preparers (`bundestag_qwen15b_lora_mps`, `bundestag_tiktoken`,
    `speakger/preparer.py`): Deterministic validation branches are missing.

- **Moderate deterministic gaps**
  - `ml_playground/sampling/runner.py` (80.31%): File-based prompt ingestion and
    compile hooks need targeted mocks.
  - `ml_playground/training/loop/runner.py` (81.08%): Best-checkpoint updates
    and evaluation-only mode are partially covered; expand fake dependency tests.

- **Stochastic or hardware-sensitive gaps**
  - `ml_playground/models/core/inference.py` (56.14%): GPU and AMP toggles need
    deterministic seeds and CPU pathways.
  - `ml_playground/training/ema.py` (40.00%): EMA decay on CUDA should be backed

## Milestones

1. **Coverage scoreboard & ownership**
   - Generate a per-module summary from `.cache/coverage/coverage.json`.
   - Current modules below 90% line coverage (needs task tracking):
     - `ml_playground/experiments/bundestag_qwen15b_lora_mps/preparer.py` — 15.52%
     - `ml_playground/experiments/bundestag_tiktoken/preparer.py` — 27.50%
     - `ml_playground/data_pipeline/preparer.py` — 33.66%
     - `ml_playground/training/ema.py` — 40.00%
     - `ml_playground/experiments/speakger/preparer.py` — 53.85%
     - `ml_playground/models/core/inference.py` — 56.14%
     - `ml_playground/cli.py` — 73.62%
     - `ml_playground/core/tokenizer_protocol.py` — 75.00%
     - `ml_playground/experiments/bundestag_char/preparer.py` — 75.86%
     - `ml_playground/core/logging_protocol.py` — 76.47%
     - `ml_playground/training/hooks/logging.py` — 76.92%
     - `ml_playground/training/hooks/runtime.py` — 78.57%
     - `ml_playground/experiments/speakger/sampler.py` — 79.51%
     - `ml_playground/configuration/loading.py` — 80.00%
   - File tasks for each module below 100%, assigning owners and capturing required DI or
     fixture work.
   - Enforce that new/changed modules reach 100% before merge.

2. **Core deterministic modules**
   - Methodically close gaps in modules with deterministic logic (`ml_playground/cli.py`,
     `configuration/loading.py`, `data_pipeline/preparer.py`).
   - Lean on existing fakes/DI seams; avoid introducing test-only branches.
{{ ... }}

3. **Experiment & runtime surfaces**
   - Cover experiment preparers, sampling runner branches, and training loop fallbacks using
     deterministic fixtures, tmp resources, and dependency injection.
   - Keep property-based suites within guideline runtime budgets while increasing branch coverage.
   - Aim for ≥99% global coverage once these modules are addressed.

4. **Stretch modules & full compliance**
   - Resolve remaining low-coverage areas (EMA, inference edge cases) via deterministic CPU
     equivalence tests and seeded runs.
   - Once per-module coverage reads 100%, raise all gates (pre-commit, CI, badges) accordingly.
   - Maintain documentation/tests to keep coverage at 100% for all future changes.

## Threshold management

- When deterministic core modules are complete (Milestone 2), raise
  `.githooks/.pre-commit-config.yaml` `--fail-under` to 95%.
- After experiment/runtime modules reach near-complete coverage (Milestone 3), raise the
  threshold to 99% and introduce per-module checks via `coverage json` analysis scripts.
- Upon full compliance (Milestone 4), set all gates to 100% line and branch coverage and keep
  them there for future work.
- Record threshold changes and milestone status in `.ldres/tv-tasks.md` and share
  progress during stand-ups.

## Tracking & ownership

- Primary DRI: Thomas (tv)
- Status updates logged in `.ldres/tv-tasks.md` under `tv-2025-10-03:PR?? · Coverage roadmap towards ~100%`.
- Related initiatives: Regression suite (tv-2025-10-03:PR??) and mutation testing (tv-2025-10-03:PR??).

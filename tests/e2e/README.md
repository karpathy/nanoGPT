# End-to-End (E2E) Tests

E2E tests exercise the application via public entry points (usually the CLI) in a realistic, but tiny, environment. They validate wiring across modules, configuration loading/merging, logging, and basic I/O.

## Scope
- Validate CLI behavior in `ml_playground/cli.py` end-to-end.
- Verify config precedence and path resolution with experiment defaults.
- Smoke-check tiny training/sampling flows without heavy models or datasets.

## Configuration
- Default E2E config: `tests/e2e/ml_playground/experiments/test_default_config.toml`
- Pass explicitly via CLI: `--exp-config tests/e2e/ml_playground/experiments/test_default_config.toml`
- Environment overrides supported by tests:
  - `ML_PLAYGROUND_TRAIN_OVERRIDES` (JSON)
  - `ML_PLAYGROUND_SAMPLE_OVERRIDES` (JSON)

## Logging
- CLI configures console logging at INFO by default (status lines printed via `logging`).
- Status helper: `_log_command_status()` in `ml_playground/cli.py` prints existence/contents of key paths.

## Principles
- Small, deterministic, fast (< seconds per test).
- No test-only branches in production code.
- Filesystem writes go to temp dirs or `out_dir` under a temporary workspace.
- Use the tiny test defaults; do not hit network or large downloads.

## Run Locally
- Run all E2E tests: `make e2e`
- Single file: `make test-file FILE=tests/e2e/path/to/test_*.py`
- Verbose logs: append `-s -vv` via `PYTEST_ADDOPTS`, e.g., `PYTEST_ADDOPTS="-s -vv" make e2e`.

## For AI Agents
- Never assume global state. Always pass `--exp-config` for E2E CLI tests.
- Prefer tiny values; keep runtime under a few seconds.
- Avoid changing production defaults for test needs; use the test config or env overrides.
- If editing CLI behavior, update E2E tests to reflect required flags/status outputs.

## Do / Don’t
- Do: test realistic CLI invocations, config merges, and log outputs.
- Do: use temporary dirs and the E2E test defaults.
- Don’t: add heavy datasets/models or network calls.
- Don’t: couple E2E tests to unit test fixtures.

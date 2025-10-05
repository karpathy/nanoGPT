# End-to-End (E2E) Tests

E2E tests exercise the application via public entry points (usually the CLI) in a realistic, but tiny, environment.
They validate wiring across modules, configuration loading/merging, logging, and basic I/O.

## Principles

- Small, deterministic, fast (\< seconds per test).
- No test-only branches in production code.
- Filesystem writes go to temp dirs or `out_dir` under a temporary workspace.
- Use the tiny test defaults; do not hit network or large downloads.

## Run Locally

- Run all E2E tests: `make e2e`
- Single file: `make test-file FILE=tests/e2e/path/to/test_*.py`
- Verbose logs: append `-s -vv` via `PYTEST_ADDOPTS`, e.g., `PYTEST_ADDOPTS="-s -vv" make e2e`.

## Folder structure

```text
tests/e2e/
├── README.md                - scope, patterns, and how to run E2E tests
├── conftest.py              - E2E pytest setup and markers
└── ml_playground/           - E2E-specific helpers and tests
    ├── experiments/         - tiny configs/fixtures for E2E
    ├── test_sample_smoke.py - CLI sampling smoke test
    └── test_train_smoke.py  - CLI training smoke test
```

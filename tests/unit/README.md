# Unit Tests

Unit tests validate individual functions, classes, and small modules in isolation. They should be trivial to read and fast to run.

## Principles

- Extremely fast, deterministic, and isolated.
- No I/O or network by default; use pure functions where possible.
- No test-specific branches in production code.

## Run Locally

- Run all unit tests: `make unit`
- Unit with coverage: `make unit-cov`
- Single file: `make test-file FILE=tests/unit/path/to/test_*.py`

## Folder structure

```text
tests/unit/
├── README.md                 - scope and rules for unit tests
├── ml_playground/            - unit tests per module/package
├── test_public_api_policy.py - enforcement of public API policy
└── conftest.py               - unit pytest fixtures and helpers

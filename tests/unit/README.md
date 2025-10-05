# Unit Tests

Unit tests validate individual functions, classes, and small modules in isolation. They should be trivial to read and fast
to run.

## Principles

- Extremely fast, deterministic, and isolated.
- No I/O or network by default; use pure functions where possible.
- No test-specific branches in production code.

## Fixtures & collaborators

- Keep unit tests self-contained. Prefer inline stub classes or dependency injection instead of monkeypatching or
  mocking.
- When shared setup is unavoidable, reuse fixtures defined in `tests/conftest.py` or a package-local `conftest.py`;
  keep them pure and deterministic.
- Follow the canonical guidance in `.dev-guidelines/TESTING.md#fixtures-strict-usage` for scope and purity rules.

## Naming

- **File names**: `test_<module>.py` within the corresponding directory (e.g.,
  `tests/unit/training/checkpointing/test_service.py`).
- **Test functions**: Prefer `test_<unit_of_behavior>_<expected_outcome>` (snake_case, verbs included when meaningful).
- **Helpers/stubs**: Prefix with `_Stub` or `_make_` to signal test-only collaborators and avoid collisions with
  production symbols.

## Testing Approaches

### Traditional Unit Tests

Standard unit tests that validate specific behaviors with hand-crafted examples.

### Property-Based Tests

Property-based tests using Hypothesis complement example-driven unit tests. They are organized separately under
`tests/property/`; see that folder's `README.md` for structure and guidelines.

## Run Locally

- Run all unit tests: `make unit`
- Unit with coverage: `make unit-cov`
- Single file: `make test-file FILE=tests/unit/path/to/test_*.py`

## Folder structure

```text
tests/unit/
├── README.md                       - scope and rules for unit tests
├── analysis/                       - analysis-related unit tests
├── configuration/                  - configuration models and loading
├── core/                           - core utilities (tokenizer, error handling)
├── data_pipeline/                  - data sources/transforms/sampling/preparer
├── experiments/                    - experiment-specific unit tests
├── sampling/                       - inference and sampling runner
├── training/                       - training loop, hooks, checkpointing, schedulers
├── test_public_api_policy.py       - enforcement of public API policy
└── conftest.py                     - unit pytest fixtures and helpers
```

# Unit Tests

Unit tests validate individual functions, classes, and small modules in isolation. They should be trivial to read and fast to run.

## Principles

- Extremely fast, deterministic, and isolated.
- No I/O or network by default; use pure functions where possible.
- No test-specific branches in production code.

## Testing Approaches

### Traditional Unit Tests
Standard unit tests that validate specific behaviors with hand-crafted examples.

### Property-Based Tests
Property-based tests using Hypothesis to validate invariants across a wide range of generated inputs:

- **Configuration System**: Tests dictionary merging, TOML serialization, and path computation with generated data structures
- **Data Loading Logic**: Tests batch sampling, memory mapping, and device placement with various array sizes and configurations

Property-based tests help catch edge cases that traditional unit tests might miss by testing against thousands of generated examples.

## Run Locally

- Run all unit tests: `make unit`
- Unit with coverage: `make unit-cov`
- Single file: `make test-file FILE=tests/unit/path/to/test_*.py`

## Folder structure

```text
tests/unit/
├── README.md                       - scope and rules for unit tests
├── analysis/                       - analysis-related unit tests
│   └── analysis/                   - LIT integration, sample quality
├── configuration/                  - configuration models and loading
├── core/                           - core utilities (tokenizer, error handling)
├── data_pipeline/                  - data sources/transforms/sampling/preparer
├── experiments/                    - experiment-specific unit tests
├── sampling/                       - inference and sampling runner
├── training/                       - training loop, hooks, checkpointing, schedulers
├── test_public_api_policy.py       - enforcement of public API policy
└── conftest.py                     - unit pytest fixtures and helpers

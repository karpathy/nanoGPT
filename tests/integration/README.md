# Integration Tests

<details>
<summary>Related documentation</summary>

- [Documentation Guidelines](../../.dev-guidelines/DOCUMENTATION.md) – Unified standards for all repository docs, covering top-level, module, experiment, test, and tool content.
- [Testing Standards](../../.dev-guidelines/TESTING.md) – Strict TDD workflow and ultra-strict testing policy for every suite.
- [Unit Tests README](../unit/README.md) – Unit tests validate individual functions, classes, and small modules in isolation.
- [Property-Based Tests README](../property/README.md) – Property-based tests validate invariants across large input spaces using Hypothesis.
- [E2E Tests README](../e2e/README.md) – E2E tests exercise public entry points in realistic, tiny environments.
- [Top-level Tests README](../README.md) – High-level testing policy, coverage guidance, and entry points for each suite.

</details>

Integration tests verify that multiple components work together correctly via Python APIs (not necessarily the CLI).
They use real code paths and small in-memory or tiny on-disk data.

## Principles

- Fast and deterministic.
- Minimal I/O; no network.
- Avoid side effects outside temp dirs.
- No test-only code paths in production.

## Run Locally

- Run all integration tests: `uvx --from . dev-tasks integration`
- Single file: `uv run pytest tests/integration/test_*.py`

## Folder structure

```text
tests/integration/
├── README.md                    - scope and patterns for integration tests
├── conftest.py                  - integration pytest setup/markers
├── test_datasets_shakespeare.py - integration of dataset helpers
└── test_speakger_pilot.py       - integration around SpeakGer pipeline
```

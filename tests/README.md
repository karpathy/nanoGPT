# Test Suites

<details>
<summary>Related documentation</summary>

- [Documentation Guidelines](../.dev-guidelines/DOCUMENTATION.md) – Unified standards for all documentation in this repository, covering top-level, module, experiment, test, and tool content.
- [Testing Standards](../.dev-guidelines/TESTING.md) – Strict TDD workflow for all functional changes with 100% success requirements across suites.
- [Unit Tests README](unit/README.md) – Unit tests validate individual functions, classes, and small modules in isolation.
- [Property-Based Tests README](property/README.md) – Property-based tests validate invariants across large input spaces using Hypothesis.
- [Integration Tests README](integration/README.md) – Integration tests verify that multiple components work together via Python APIs using real code paths and tiny data.
- [E2E Tests README](e2e/README.md) – E2E tests exercise the application via public entry points in realistic, tiny environments.

</details>

The `tests/` directory hosts all automated checks. Each subfolder documents its own
scope; this top-level README provides the high-level testing policy and entry points.

## Structure

```text
tests/
├── README.md            - this file
├── acceptance/          - policy and workflow enforcement via CLI
├── conftest.py          - shared fixtures limited to stable, deterministic helpers
├── e2e/                 - end-to-end CLI smoke tests
├── integration/         - multi-module behaviors using public APIs
├── property/            - Hypothesis properties scoped as part of coverage gates
├── support/             - shared data/assets for tests (read-only)
└── unit/                - exemplar-driven unit tests (fast, deterministic)
```

## Coverage Policy

- `uvx --from . dev-tasks coverage-report` runs `pytest tests/unit tests/property` under coverage.
- Integration, E2E, and acceptance suites do not contribute to coverage gates, but they
  still run in CI.
- See `.dev-guidelines/TESTING.md` for thresholds and gating rules.

## Running Tests

- **Fast feedback (unit + property)**: `uvx --from . dev-tasks coverage-report`
- **Specific suite**: `uv run pytest tests/<suite>/`
- **Single file**: `uv run pytest tests/<suite>/path/to/test_file.py`
- **Acceptance workflows**: `uvx --from . dev-tasks acceptance`

## Additional References

- `.dev-guidelines/TESTING.md` – authoritative testing policies.
- `tests/unit/README.md` – detailed unit-test conventions.
- `tests/property/README.md` – Hypothesis guidance and folder layout.

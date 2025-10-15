# Test Suites

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
    └── <package>/       - mirrors `src/ml_playground/<package>/` for unit tests
```

Unit tests always live under `tests/unit/<package>/...`, mirroring the namespace
in `src/ml_playground/`. Do not create alternative layouts such as
`tests/ml_playground/unit`; keep the suite hierarchy stable so imports, tooling,
and documentation remain aligned with the developer guidelines
(`.dev-guidelines/TESTING.md`).

## Coverage Policy

- `uv run ci-tasks coverage-report` runs `pytest tests/unit tests/property` under coverage.
- Integration, E2E, and acceptance suites do not contribute to coverage gates, but they
  still run in CI.
- See `.dev-guidelines/TESTING.md` for thresholds and gating rules.

## Running Tests

- **Fast feedback (unit + property)**: `uv run ci-tasks coverage-report`
- **Specific suite**: `uv run pytest tests/<suite>/`
- **Single file**: `uv run pytest tests/<suite>/path/to/test_file.py`
- **Acceptance workflows**: `uv run test-tasks acceptance`

## Additional References

- `.dev-guidelines/TESTING.md` – authoritative testing policies.
- `tests/unit/README.md` – detailed unit-test conventions.
- `tests/property/README.md` – Hypothesis guidance and folder layout.

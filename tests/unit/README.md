# Unit Tests

Unit tests validate individual functions, classes, and small modules in isolation. They should be trivial to read and fast to run.

## Scope
- Single unit at a time, with dependencies mocked or stubbed as needed.
- No CLI or filesystem unless the unit is specifically about those.
- No reliance on external TOML defaults.

## Configuration
- Construct config dataclasses/Pydantic models inline with tiny values.
- Avoid loading files; prefer direct objects and minimal test data.

## Principles
- Extremely fast, deterministic, and isolated.
- No I/O or network by default; use pure functions where possible.
- No test-specific branches in production code.

## Run Locally
- Run all unit tests: `uv run pytest -q tests/unit`
- Single file: `uv run pytest -q tests/unit/path/to/test_*.py`

## For AI Agents
- Prefer direct calls to functions/classes with small inputs.
- Use mocks to isolate behavior and avoid global state.
- Keep tests independent of integration/E2E fixtures and configs.

## Do / Don’t
- Do: assert precise behavior of a single unit.
- Do: mock neighbors and keep tests minimal.
- Don’t: import or depend on E2E defaults or CLI wiring.
- Don’t: rely on filesystem or environment unless the unit under test requires it.

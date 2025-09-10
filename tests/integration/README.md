# Integration Tests

Integration tests verify that multiple components work together correctly via Python APIs (not necessarily the CLI). They use real code paths and small in-memory or tiny on-disk data.

## Scope
- Validate cross-module behavior (e.g., data transforms, sampling pipelines).
- Use Python entry points instead of CLI when feasible.
- No reliance on E2E-specific defaults; configs are provided inline or via small helpers.

## Configuration
- Prefer explicit, minimal config objects in tests.
- If files are needed, write tiny TOML/CSV/JSON fixtures to temp dirs.
- Do not rely on `tests/e2e/.../test_default_config.toml`.

## Principles
- Fast and deterministic.
- Minimal I/O; no network.
- Avoid side effects outside temp dirs.
- No test-only code paths in production.

## Run Locally
- Run all integration tests: `make integration`
- Single file: `make test-file FILE=tests/integration/test_*.py`

## For AI Agents
- Use explicit Python configs and small, local fixtures.
- Keep runtime small; prefer pure-Python or tiny files.
- Maintain isolation: each test should set up and tear down its own state.

## Do / Don’t
- Do: compose real modules and check behavior through their public APIs.
- Do: keep fixtures tiny and self-contained.
- Don’t: call the CLI unless you’re validating CLI wiring (that belongs to E2E).
- Don’t: depend on E2E defaults or large resources.

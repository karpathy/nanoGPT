# Unit Tests

Unit tests validate individual functions, classes, and small modules in isolation. They should be trivial to read and fast to run.

## Scope
- Single unit at a time, with dependencies mocked or stubbed as needed.
- No CLI or filesystem unless the unit is specifically about those.
- No reliance on external TOML defaults.

## Configuration
- Construct config dataclasses/Pydantic models inline with tiny values.
- Avoid loading files; prefer direct objects and minimal test data.

### Centralized fixtures and helpers

- `tests/conftest.py`
  - `out_dir(tmp_path: Path) -> Path`
    - Provides a ready-to-use `tmp_path / "out"` directory for tests that write outputs.
  - `minimal_full_experiment_toml(dataset_dir: Path, out_dir: Path, *, extra_optim: str = "", extra_train: str = "", extra_sample: str = "", extra_sample_sample: str = "", include_train_data: bool = True, include_train_runtime: bool = True, include_sample: bool = True) -> str`
    - Builds a strict, minimal ExperimentConfig TOML. Use the `extra_*` parameters to inject lines into the appropriate sections and the `include_*` flags to omit sections when testing validation errors.

- `tests/unit/ml_playground/conftest.py`
  - `ListLogger` with fixtures `list_logger` and `list_logger_factory` to capture `.info()` and `.warning()` calls without writing to stdout.

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

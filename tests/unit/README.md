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

## Public API Only

All unit tests must target only the public interfaces of production modules. Do not import or call private helpers (underscore-prefixed names) or reach into internal submodules that are not part of the documented surface.

Allowed canonical imports:

- `from ml_playground import config` (e.g., `DataConfig`, `TrainerConfig`, `RuntimeConfig`, `load_experiment_toml`)
- `import ml_playground.config_loader as config_loader` (e.g., `load_full_experiment_config`, `deep_merge_dicts`)
- `import ml_playground.prepare as prepare` (e.g., `make_preparer`, `write_bin_and_meta`, `prepare_with_tokenizer`)
- `import ml_playground.trainer as trainer`
- `import ml_playground.sampler as sampler`
- `from ml_playground.analysis import run_server_bundestag_char`
- `import ml_playground.cli as cli` and use `cli.app` or `cli.main()` for CLI tests via `CliRunner`.

Forbidden patterns (examples):

- `from ml_playground.<module> import _something` (private underscore-prefixed symbols)
- `from ml_playground.analysis.lit_integration import ...` (use `from ml_playground.analysis import ...`)
- `from ml_playground.config import _deep_merge_dicts` (use `config_loader.deep_merge_dicts`)
- `ml_playground.cli._run_*`, `ml_playground.cli._global_device_setup` (exercise behavior via CLI commands)

If a test requires functionality that only exists as a private helper, prefer either using a public entrypoint that exercises the same behavior or proposing a thin public façade in production code that delegates to the private implementation.

## Run Locally

- Run all unit tests: `make unit`
- Unit with coverage: `make unit-cov`
- Single file: `make test-file FILE=tests/unit/path/to/test_*.py`

## For AI Agents

- Prefer direct calls to functions/classes with small inputs.
- Use mocks to isolate behavior and avoid global state.
- Keep tests independent of integration/E2E fixtures and configs.

## Do / Don’t

- Do: assert precise behavior of a single unit.
- Do: mock neighbors and keep tests minimal.
- Don’t: import or depend on E2E defaults or CLI wiring.
- Don’t: rely on filesystem or environment unless the unit under test requires it.

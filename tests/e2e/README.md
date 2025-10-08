# End-to-End (E2E) Tests

<details>
<summary>Related documentation</summary>

- [Documentation Guidelines](../../.dev-guidelines/DOCUMENTATION.md) – Unified standards for all repository docs, covering top-level, module, experiment, test, and tool content.
- [Testing Standards](../../.dev-guidelines/TESTING.md) – Strict TDD workflow and ultra-strict testing policy for every suite.
- [Unit Tests README](../unit/README.md) – Unit tests validate individual functions, classes, and small modules in isolation.
- [Property-Based Tests README](../property/README.md) – Property-based tests validate invariants across large input spaces using Hypothesis.
- [Integration Tests README](../integration/README.md) – Integration tests verify multi-component interactions via Python APIs.
- [Top-level Tests README](../README.md) – High-level overview of the testing tree, policies, and entry points.

</details>

E2E tests exercise the application via public entry points (usually the CLI) in a realistic, but tiny, environment.
They validate wiring across modules, configuration loading/merging, logging, and basic I/O.

## Principles

- Small, deterministic, fast (\< seconds per test).
- No test-only branches in production code.
- Filesystem writes go to temp dirs or `out_dir` under a temporary workspace.
- Use the tiny test defaults; do not hit network or large downloads.

## Configuration Guidelines

- Keep datasets synthetic and reusable. `tests/e2e/ml_playground/experiments/bundestag_char/test_cli_bundestag_char.py`
  writes a 512-token uint16 corpus so both `train.bin` and `val.bin` satisfy CLI preconditions while still
  fitting in memory and keeping sampling deterministic.
- Choose the smallest model/training knobs that still trigger the behavior under test. The bundestag-char
  helper `_write_exp_config()` pins `n_layer=1`, `n_head=2`, `n_embd=64`, `block_size=64`, `batch_size=4`,
  and `max_iters=4` so the CLI creates checkpoints (`keep.last=1`, `keep.best=1`) and emits `meta.pkl` in
  under 0.1 seconds. Smaller values would fail runtime validation (e.g., `block_size` vs dataset length) or
  skip checkpoint rotation, weakening coverage.
- Reuse artifacts when chaining commands. `test_sample_bundestag_char_quick` trains once, then points the
  sample command at the same out directory to avoid redundant work while verifying both CLI pathways.
- Prefer dependency injection over heavyweight downloads. `tests/e2e/ml_playground/experiments/speakger/test_sampler_analysis.py`
  supplies `DummyTokenizer`, `DummyBaseModel`, and `DummyPeftModel` factories so the sampler produces a
  single JSON/text pair without touching external checkpoints. Its `SamplerConfig` limits work to
  `num_samples=1` and `max_new_tokens=5`, enough to exercise formatting and analysis logic.
- Document any special fixtures or magic numbers directly in the corresponding test helpers so future
  contributors understand why those values were chosen and what regressions they guard against.

## Run Locally

- Run all E2E tests: `uvx --from . dev-tasks e2e`
- Single file: `uv run pytest tests/e2e/path/to/test_*.py`
- Verbose logs: append `-s -vv` via `PYTEST_ADDOPTS`, e.g., `PYTEST_ADDOPTS="-s -vv" uvx --from . dev-tasks e2e`.

## Folder structure

```text
tests/e2e/
├── README.md                - scope, patterns, and how to run E2E tests
├── conftest.py              - E2E pytest setup and markers
└── ml_playground/           - E2E-specific helpers and tests
    ├── experiments/         - tiny configs/fixtures for E2E
    ├── test_sample_smoke.py - CLI sampling smoke test
    └── test_train_smoke.py  - CLI training smoke test
```

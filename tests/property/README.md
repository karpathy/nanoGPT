# Property-Based Tests

Property-based tests validate invariants across large input spaces using Hypothesis.
They run alongside unit tests but live in this dedicated suite so Hypothesis-specific
configuration stays isolated.

## Principles

- Keep properties deterministic: set explicit `@settings(derandomize=True)` or pin the
  Hypothesis seed via environment variables.
- Use dependency injection seams (e.g., `CLIDependencies`, configuration factories)
  instead of monkeypatching or mocks.
- Prefer `TemporaryDirectory()` or other context managers over function-scoped fixtures.
- Exercise public entry points only, per `.dev-guidelines/TESTING.md#public-vs-private-apis`.

## Folder Structure

```text
tests/property/
├── README.md                       - this file
├── cli/                            - CLI-facing properties
├── configuration/                  - TOML loading and config invariants
└── data_pipeline/                  - data preparation/tokenization properties
```

## Running

- Full property suite (with unit tests): `make coverage-report`
- Specific property module: `uv run pytest tests/property/<path>/test_*.py`

## Capturing Shrunk Examples as Regression Tests

- **Trigger**: Let Hypothesis shrink a failing input (store is under `.cache/hypothesis/`).
- **Inspect**: Run `uv run python -m hypothesis show <test-module>::<test-name>` to print the shrunken case if available, or open the cached JSON in `.cache/hypothesis/`.
- **Codify**: Translate the minimal input into a deterministic check using `@example(...)` or an explicit unit/property test. Prefer fixtures/helpers over hard-coded globals.
- **Verify**: Rerun the relevant module (`uv run pytest tests/property/<path>/test_*.py`) to ensure the new guardrails fail without the fix and pass with it.
- **Document**: Leave a brief comment referencing the original failure or issue to aid future triage.

## Related Documentation

- `.dev-guidelines/TESTING.md` – Hypothesis guidance, coverage gates, fixture rules.
- `tests/unit/README.md` – complementary unit-test conventions.

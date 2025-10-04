# Testing Standards

## Test-Driven Development (REQUIRED)

We practice strict TDD for all functional changes:

1. Write a failing test that specifies the intended behavior (unit or integration).
2. Implement the minimal production code to make the test pass.
3. Refactor safely with tests green.

Commit strategy under TDD:

- Prefer small commits per behavior
- The implementation commit MUST pair production code and its tests if the previous commit wasn't already pairing them.
- Never merge code that reduces coverage or leaves failing tests.

Allowed deviations:

- Documentation-only changes.
- Test-only refactors (no behavior change) when clearly labeled `test(<scope>): ...`.
- Mechanical format/lint fixes.

## Testing Standards (ULTRA-STRICT POLICY - 100% SUCCESS REQUIRED)

Testing Docs

- Unit tests: see `tests/unit/README.md` (isolation, speed, pure-Python focus)
- Integration tests: see `tests/integration/README.md` (compose small real components)
- End-to-end (E2E): see `tests/e2e/README.md` (CLI wiring, config/paths, logging)
  - For CLI-based E2E tests, pass the tiny defaults explicitly:
    `--exp-config tests/e2e/ml_playground/experiments/test_default_config.toml`

### 1. Test Framework and Runner

- **Framework**: pytest only. Do not use unittest or nose.
- **Runner**: `make test` (invokes pytest under the hood)
- **Coverage**: `make coverage-badge`
  (generates reports under `.cache/coverage` and refreshes
  `docs/assets/coverage.svg`, `docs/assets/coverage-lines.svg`, and
  `docs/assets/coverage-branches.svg`)
- **Random seed**: Enforced determinism via `tests/conftest.py` with fixed seed.

**Rationale**: One toolchain avoids fragmentation and flakiness.

### 2. Test Levels (ULTRA-STRICT Performance Requirements)

- **Unit tests** (required): LIGHTNING FAST (<10ms each), isolated, no network, no filesystem writes. One spec per
  behavior. MUST achieve 100% success rate.
- **Integration tests** (allowed/required): Minimal real integration across 2–3 components, no live external services.
  Use in-memory or ephemeral resources. Must complete in <100ms each.
- **End-to-end tests** (discouraged): Only when explicitly approved; must run in CI under 30s total with
  recorded/replayed I/O.

**Rationale**: Speed is critical for developer productivity; any slow test breaks the flow.

### 3. Directory Layout and Naming

- **Structure** (canonical): `tests/unit/<package>/test_<module>.py`
- **Packages** mirror `ml_playground/` layout, for example:
  - `tests/unit/training/`, `tests/unit/sampling/`, `tests/unit/data_pipeline/`, `tests/unit/configuration/`,
    `tests/unit/core/`, `tests/unit/experiments/`, `tests/unit/analysis/`
- **Test functions**: `test_<behavior>_<condition>_<expected>()`
- **Test classes** (if grouping needed): `Test<Subject>` only; no `__init__` in test classes.
- **Docstrings**: Each test function must have a one-line docstring stating the behavior it covers.

Quick reference (examples):

```text
ml_playground/training/checkpointing/     -> tests/unit/training/checkpointing/test_<module>.py
ml_playground/sampling/                   -> tests/unit/sampling/test_<module>.py
ml_playground/data_pipeline/              -> tests/unit/data_pipeline/test_<module>.py
ml_playground/configuration/              -> tests/unit/configuration/test_<module>.py
ml_playground/models/                     -> tests/unit/core/test_<module>.py
```

**Rationale**: Predictable discovery and easy navigation.

### 4. What to Test (Scope Rules)

- **Public API**: Must be covered.
- **Bug fixes**: Add failing test first; keep it.
- **Branches**: Each logical branch, error path, and exception message for public functions.
- **Types and contracts**: If type hints exist, test boundary values and None handling.

**Rationale**: Tests document contracts and prevent regressions.

### 4.1 Property-Based vs. Example Unit Tests

- **Framework**: Use Hypothesis for property-based testing.
  Leverage `@example(...)` to encode canonical cases alongside generated data.
- **When to favor properties**:
  - Validate algebraic laws, round-trips, invariants, and metamorphic relations for pure, deterministic code paths.
  - Explore broad input spaces for serializers, parsers, math/algorithmic utilities, and transformations.
  - Capture bug reproductions: promote shrunk counterexamples into `@example(...)` inputs.
- **Keep classic example/unit tests when**:
  - Document named business rules or requirements.
    (Keep `test_orders_over_100_get_free_shipping()` as a standalone example for clarity.)
  - Guard regressions tied to a specific defect; the test name should reference the bug or issue.
  - Exercise integration seams (HTTP, DB, time, concurrency) where properties lack a precise oracle.
  - Assert protocol/state-machine flows or golden outputs where readability matters more than exhaustive exploration.
- **Combining approaches**:
  - Prefer a property plus a minimal set of named examples instead of parallel example tests covering the same logic.
  - Stabilize CI by enabling `derandomize=true` (or fixed seeds).
    Set explicit `@settings(max_examples=..., deadline=...)` to control runtime.
  - Rotate seeds locally when hunting for new inputs; commit only deterministic seeds/examples.
- **Placement**: Store property suites in files ending with `_property.py`
  (e.g., `test_config_property.py`). Example-only suites remain in standard
  `test_<module>.py` files.
- **Performance**: Enforce strict time budgets. Tune `max_examples` and `deadline`
  so each property completes comfortably within the unit-test thresholds.
- **Determinism**: Property-based tests must run deterministically under CI seeds
  and fail with the shrunk example in their assertion message.

**Rationale**: Properties deliver broad coverage for pure logic, while focused
example tests preserve readable specifications for business rules, regressions,
and messy boundaries.

### 5. Test Writing Style

- **AAA structure**: Arrange, Act, Assert; one assert type per test unless parametrized.
- **Parametrize**: Use `@pytest.mark.parametrize` instead of loops/if-else in tests.
- **No logic**: Beyond simple comprehensions and literals.
- **Explicit values**: Avoid magic numbers—name them.

**Rationale**: Readable tests fail clearly and localize faults.

### 5.1 No Test-Specific Code Paths in Production (Non-Negotiable)

- Production code must never contain branches, flags, or behavior that exists only to satisfy tests.
  - Examples of forbidden patterns: `if TESTING: ...`, checking `PYTEST_CURRENT_TEST`, special test-only parameters,
    alternate I/O paths only under tests.
- Tests must exercise the same public API and code paths used in production.
- Make code testable via proper seams instead:
  - Dependency injection with sensible production defaults (e.g., pass Path, clock, RNG, HTTP client).
  - Use pytest fixtures and mocks/monkeypatch for external boundaries (network, filesystem, time, env).
- Idempotency and determinism are product qualities, not test toggles. Implement them unconditionally where
  applicable.

### 6. Fixtures (Strict Usage)

- **Scope**: Prefer function-scoped; module/session only for expensive immutable data.
- **Location**: All shared setup in `tests/conftest.py`.
- **Purity**: Fixtures must be pure and return simple objects; avoid side effects.
- **I/O fixtures**: If writing to disk/network, must be session-scoped, opt-in, behind marker.

**Rationale**: Contained, fast, and reproducible tests.

### 7. Mocking and Fakes (No Exceptions)

- **No mocking or monkeypatching anywhere** (unit, integration, or E2E).
  This includes `pytest.monkeypatch`, `unittest.mock`, `pytest_mock`, and
  similar APIs.
- **Use dependency injection and fakes** exclusively:
  - Provide lightweight in-memory fakes and DI seams in production code where
    collaboration is required.
  - For external boundaries (network, filesystem, time, randomness,
    environment, subprocess), use deterministic fakes, tmp resources, or
    seamable adapters.
- **No live HTTP**: Use fake clients/adapters with recorded deterministic
  responses.

Enforcement:

- A pre-commit hook forbids the following tokens anywhere under `tests/`:
  `monkeypatch`, `pytest.MonkeyPatch`, `unittest.mock`,
  `from unittest import mock`, `pytest_mock`, `MagicMock`, `patch(`.
- Ruff banned-API also flags disallowed imports/usages in code.

**Rationale**: Determinism, stability, and meaningful coverage without runtime
patching.

### 8. Data and I/O

- **Default**: No filesystem writes. If necessary, use `tmp_path` fixture only.
- **Test data**: Lives in `tests/data/`; small, versioned, deterministic. Use CSV/JSON/YAML.
- **Data creation**: For pandas/numpy, create inline for clarity unless size requires files.

**Rationale**: Deterministic, reviewable inputs.

### 9. Randomness, Time, and Concurrency

- **Randomness**: Seeded via `conftest.py`; inject seeds via parameters where possible.
- **Time**: Freeze time via monkeypatching; never rely on real clocks for assertions.
- **Concurrency**: Avoid sleeps. Use synchronization primitives or polling with short timeout.

**Rationale**: Flakes come from nondeterminism and timing.

### 10. Test Markers and Selection

- **Allowed markers**: `slow`, `integration`, `perf`
- **CI default**: Exclude `slow` and `perf`; include `integration`
- **Developer default**: Run everything except `perf`

**Rationale**: Fast feedback loop with targeted opt-ins.

### 11. Coverage Requirements (ULTRA-STRICT CI Gates)

- **Global line coverage**: 100% (NO EXCEPTIONS)
- **Per-module line coverage**: 100% for ALL `ml_playground/*` modules
- **Branch coverage**: 100% for ALL modules (NO COMPROMISES)
- **New/changed code**: Must achieve 100% coverage before merge
- **No pragma comments**: ABSOLUTELY FORBIDDEN except for impossible-to-test code (`if 0:`,
  `if __name__ == "__main__":`).

**Rationale**: Complete test coverage ensures zero blind spots and maximum confidence in code quality.

**Badge workflow**:

- Pre-commit automatically runs `make coverage-badge` and stages the refreshed line and branch badges.
- CI re-runs the same target and fails if any of the coverage badge SVGs differ from the committed versions.

### 12. Flaky Test Policy (IMMEDIATE ACTION REQUIRED)

**ABSOLUTE ZERO TOLERANCE**: Any flaky test is IMMEDIATELY removed from the suite. NO 24h grace period. NO xfail
exceptions. NO second chances.

**Enforcement**: First flake = immediate deletion. Tests must be 100% deterministic and reliable.

**Rationale**: Even a single flaky test destroys developer confidence and wastes precious development time.

## Mutation Testing (Optional)

- We use Cosmic Ray for mutation testing; configuration lives in `pyproject.toml` under `[tool.cosmic-ray]`.
- Run manually when desired:

```bash
make quality-ext
```

- This initializes (if needed) and executes Cosmic Ray sessions at `.cache/cosmic-ray/session.sqlite`. The step is
  non-fatal and not part of CI or pre-commit by default.
- The `.cache/cosmic-ray/` directory is treated like other build artifacts: never commit it, clean it with `make clean`
  if needed.

## Running Tests

### Local Development

```bash
# Fast check
make pytest-core PYARGS="-q"

# With markers (exclude slow/perf)
make pytest-core PYARGS='-m "not slow and not perf" -q'

# Full suite with coverage
make coverage
```

### CI Commands

```bash
# Coverage check (CI)
make coverage
```

## Example Test Patterns

### Parametrized Unit Test

```python
from __future__ import annotations
import pytest
from pathlib import Path
from ml_playground.config import load_toml

@pytest.mark.parametrize(
    "block_size,expected_valid",
    [
        (16, True),
        (32, True),
        (0, False),
        (-1, False),
    ],
)
def test_config_validates_block_size(tmp_path: Path, block_size: int, expected_valid: bool) -> None:
    """Test that config validation handles different block_size values correctly."""
    # Arrange
    config_content = f"""
[model]
block_size = {block_size}
n_layer = 1
n_head = 1
n_embd = 32
vocab_size = 256
"""
    config_path = tmp_path / "test.toml"
    config_path.write_text(config_content, encoding="utf-8")

    # Act / Assert
    if expected_valid:
        exp = load_full_experiment_config(config_path)
        assert exp.train.model.block_size == block_size
    else:
        with pytest.raises(ValueError):
            load_full_experiment_config(config_path)

### Fixture with tmp_path

from pathlib import Path
import pytest

@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    return path
```

## Adding New Tests

Create files under `tests/unit/<package>/test_<module>.py`:

```python
from __future__ import annotations
from pathlib import Path
import pytest
from ml_playground.configuration import load_full_experiment_config


def test_config_loading_handles_missing_file(tmp_path: Path) -> None:
    """Test that config loading raises appropriate error for missing files."""
    # Arrange
    missing_path = tmp_path / "missing.toml"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_full_experiment_config(missing_path)
```

## -trigger: always_on description: Exhaustive testing policies enforcing TDD, coverage, and determinism

# Testing Standards

<details>
<summary>Related documentation</summary>

- [Developer Guidelines Index](./Readme.md) – Entry point for core policies and quick-start commands.
- [Development Practices](./DEVELOPMENT.md) – Commit pairing rules, quality gates, and tooling workflows.

</details>

## Table of Contents

- [Test-Driven Development (REQUIRED)](#test-driven-development-required)
- [Testing Standards (ULTRA-STRICT POLICY - 100% SUCCESS REQUIRED)](#testing-standards-ultra-strict-policy---100-success-required)
- [Mutation Testing (Optional)](#mutation-testing-optional)
- [Running Tests](#running-tests)
- [Example Test Patterns](#example-test-patterns)
- [Adding New Tests](#adding-new-tests)

## Test-Driven Development (REQUIRED)

We practice strict TDD for all functional changes:

1. Write a failing test that specifies the intended behavior (unit or integration).
1. Implement the minimal production code to make the test pass.
1. Refactor safely with tests green.

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
- **Runner**: `uvx --from . test-tasks test` (invokes pytest under the hood)
- **Coverage**: `uvx --from . ci-tasks coverage-badge`
  (generates reports under `.cache/coverage` and refreshes
  `docs/assets/coverage.svg`, `docs/assets/coverage-lines.svg`, and
  `docs/assets/coverage-branches.svg`)
- **Random seed**: Enforced determinism via `tests/conftest.py` with fixed seed.

**Rationale**: One toolchain avoids fragmentation and flakiness.

### 2. Test Levels (ULTRA-STRICT Performance Requirements)

- **Unit tests** (required): LIGHTNING FAST (\<10ms each), isolated, no network, no filesystem writes. One spec per
  behavior. MUST achieve 100% success rate.
- **Integration tests** (allowed/required): Minimal real integration across 2–3 components, no live external services.
  Use in-memory or ephemeral resources. Must complete in \<100ms each.
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

### 4.1 Property-Based Testing First

- **Default posture**: Start every new test effort with a Hypothesis property.
  Only add example/unit tests when a property cannot adequately encode the oracle
  or when readability of a named scenario is paramount.
- **Framework**: Hypothesis is mandatory for properties.
  Use `@example(...)` to pin canonical cases and previously discovered counterexamples.
- **Design checklist before writing a property**:
  - Identify invariants, round-trips, metamorphic relations, or conservation laws.
  - Define the input strategy (custom `@st.composite` where needed) that reflects
    production constraints.
  - Set explicit `@settings(...)` for `max_examples`, `deadline`, and
    `derandomize=True` to honor the runtime budgets in §2.
- **When example/unit tests are acceptable**:
  - Document an explicit business rule or regression where the name tells the
    story (e.g., `test_orders_over_100_get_free_shipping`).
  - Validate behavior that depends on opaque collaborators where crafting a
    deterministic property would duplicate the implementation.
  - Assert protocol/state-machine flows or golden outputs best expressed as a
    short script.
- **Hybrid approach**:
  - Pair each property with the minimum set of named examples needed for clarity.
  - Promote every shrunk counterexample to an `@example(...)` entry.
  - Avoid parallel example tests that repeat the exact logic already covered by a
    property.
- **Organization**: Place property suites in files named `test_<subject>_property.py`.
  Example-focused suites remain in `test_<module>.py`.
- **Determinism & performance**: Properties must pass deterministically under CI
  seeds. Tune strategies/settings so each property completes comfortably within
  the \<10 ms unit-test budget (or justify the overage in the test docstring).

**Rationale**: A property-first mindset maximizes behavioral exploration while
retaining example tests for narrative requirements and hard-to-oracle seams.

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

- **Scope**: Coverage gates are enforced exclusively via the unit (`tests/unit/`) and
  property-based (`tests/property/`) suites. Integration, E2E, acceptance, and perf
  suites do **not** participate in coverage runs.
- **Global line coverage**: 100% (NO EXCEPTIONS) across `ml_playground/*` when driven by
  the unit + property suites.
- **Per-module line coverage**: 100% for ALL `ml_playground/*` modules.
- **Branch coverage**: 100% for ALL modules (NO COMPROMISES).
- **New/changed code**: Must achieve 100% coverage (line + branch) under the unit +
  property suites before merge.
- **No pragma comments**: ABSOLUTELY FORBIDDEN except for impossible-to-test code (`if 0:`,
  `if __name__ == "__main__":`).

**Rationale**: Unit + property suites deliver exhaustive coverage for production code
while keeping slower integration/E2E flows outside the gating path.

**Badge workflow**:

- Pre-commit automatically runs `uvx --from . ci-tasks coverage-badge` and stages the refreshed line and branch badges.
- CI re-runs the same target and fails if any of the coverage badge SVGs differ from the committed versions.

### 12. Flaky Test Policy (IMMEDIATE ACTION REQUIRED)

**ABSOLUTE ZERO TOLERANCE**: Any flaky test is IMMEDIATELY removed from the suite. NO 24h grace period. NO xfail
exceptions. NO second chances.

**Enforcement**: First flake = immediate deletion. Tests must be 100% deterministic and reliable.

**Rationale**: Even a single flaky test destroys developer confidence and wastes precious development time.

## Mutation Testing (Optional)

- **Tooling**: Cosmic Ray configuration lives in `pyproject.toml` under `[cosmic-ray]`.

- **Primary local command**:

  ```bash
  uvx --from . ci-tasks mutation run
  ```

  This target resets `.cache/cosmic-ray/session.sqlite`, prints the active configuration via `tools/mutation_summary.py`,
  runs `cosmic-ray init/exec`, and finishes with `tools/mutation_report.py` to display survivor counts.

- **Alternative local command**: `uvx --from . ci-tasks quality-ext` executes the broader quality suite, including mutation if desired.

- **Default scope**: `module-path = "ml_playground"`, exercising the entire package with
  `pytest -q -n auto tests/unit` and a 1 s timeout per mutant/test run.

- **Latest baseline (2025-10-06)**: `uvx --from . ci-tasks mutation run` processed **5 314** mutants (killed: **5 312**, incompetent: **2**)
  in approximately **1 h 31 m** wall-clock time.

- **Session hygiene**: Both targets delete the session DB on every run; do not commit `.cache/cosmic-ray/`.

- **Follow-up**: Survivor automation and module-specific hardening tasks are tracked in `.ldres/tv-tasks.md`.

- **CI workflow**: Trigger the long-running mutation suite via GitHub Actions → *Mutation Suite* workflow. It runs weekly on Mondays at 01:00 UTC and is available on-demand through the *Run workflow* button.

- The `.cache/cosmic-ray/` directory is treated like other build artifacts: never commit it; clean with `uvx --from . env-tasks clean` if needed.

## Running Tests

Use the Typer wrappers for consistency with CI:

```bash
# Fast check
uvx --from . test-tasks pytest -- -q

# With markers (exclude slow/perf)
uvx --from . test-tasks pytest -- -m "not slow and not perf" -q

# Full suite with coverage
uvx --from . ci-tasks coverage-report --fail-under 87
```

Invoke targeted suites directly with `uv run pytest path/to/test.py` when iterating on a single module, but ensure the commands above pass before committing.

### CI Commands

```bash
# Coverage check (CI)
uvx --from . ci-tasks coverage-report --fail-under 87
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

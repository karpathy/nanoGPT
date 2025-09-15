---
trigger: always_on
description: 
globs: 
---

# ml_playground Development Practices

Core development practices, quality standards, and workflow for ml_playground contributors.

## Quality Gates (Mandatory)

For detailed information about the centralized framework utilities, see [Framework Utilities Documentation](../docs/framework_utilities.md).

Run this Make target before every commit (same commands under the hood):

```bash
make quality
```

This runs: ruff (lint+format), pyright, mypy (ml_playground), and pytest with strict settings.

CI and pre-commit both invoke `make quality` as the core gate.

## Commit Standards

### Granular Commits Policy

- **One logical change per commit** (e.g., fix a test, adjust a config, refactor a function)
- **Keep commits under ~200 lines** unless unavoidable
- **Run quality gates before each commit**, not just before PR
- **Pairing rule (REQUIRED)**: Each functional or behavioral change MUST include its tests in the same commit (unit and/or integration). Creating new files (untracked) is expected when adding tests—stage them together with the production change.
- **Granularity**: Prefer a short sequence of TDD commits to a single large commit. Each step should keep the suite green.

### Practical Tips

- Stage hunks selectively: `git add -p`
- Separate mechanical formatting from semantic changes
- Split large features into reviewable increments
- Keep tests passing at every step
- Pair production and test changes: when adding/refactoring code, include the minimal tests that specify the behavior in the same commit.
- Acceptable exceptions: documentation only commits, pure test refactors (no behavior change), or mechanical formatting. For everything else, pair code+tests.

### Conventional Commit Format

**Required format**: `<type>(<scope>): <subject>`

**Types**: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
**Scope**: module or area (e.g., trainer, config, guidelines, tests)
**Subject**: imperative, concise, lowercase (no trailing period)

**Examples**:

- `feat(trainer): write checkpoint sidecar JSON with decision inputs/outputs`
- `test(trainer): add tests for checkpoint sidecar schema and behavior`
- `feat(config): reject invalid out_dir` + tests in same commit (preferred)
- `chore(config): centralize tooling settings in pyproject.toml`
- `docs(guidelines): document pyproject-only config and granular commits + TDD`

## Test-Driven Development (REQUIRED)

We practice strict TDD for all functional changes:

1. Write a failing test that specifies the intended behavior (unit or integration).
2. Implement the minimal production code to make the test pass.
3. Refactor safely with tests green.

Commit strategy under TDD:

- Prefer small commits per behavior
- The implementation commit MUST pair production code and its tests if the previous commit wasn’t already pairing them.
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
- **Coverage**: `make coverage`
- **Random seed**: Enforced determinism via `tests/conftest.py` with fixed seed.

**Rationale**: One toolchain avoids fragmentation and flakiness.

### 2. Test Levels (ULTRA-STRICT Performance Requirements)

- **Unit tests** (required): LIGHTNING FAST (<10ms each), isolated, no network, no filesystem writes. One spec per behavior. MUST achieve 100% success rate.
- **Integration tests** (allowed/required): Minimal real integration across 2–3 components, no live external services. Use in-memory or ephemeral resources. Must complete in <100ms each.
- **End-to-end tests** (discouraged): Only when explicitly approved; must run in CI under 30s total with recorded/replayed I/O.

**Rationale**: Speed is critical for developer productivity; any slow test breaks the flow.

### 3. Directory Layout and Naming

- **Structure**: `tests/unit/ml_playground/test_<module>.py`
- **Test functions**: `test_<behavior>_<condition>_<expected>()`
- **Test classes** (if grouping needed): `Test<Subject>` only; no `__init__` in test classes.
- **Docstrings**: Each test function must have a one-line docstring stating the behavior it covers.

**Rationale**: Predictable discovery and easy navigation.

### 4. What to Test (Scope Rules)

- **Public API**: Must be covered.
- **Bug fixes**: Add failing test first; keep it.
- **Branches**: Each logical branch, error path, and exception message for public functions.
- **Types and contracts**: If type hints exist, test boundary values and None handling.

**Rationale**: Tests document contracts and prevent regressions.

### 5. Test Writing Style

- **AAA structure**: Arrange, Act, Assert; one assert type per test unless parametrized.
- **Parametrize**: Use `@pytest.mark.parametrize` instead of loops/if-else in tests.
- **No logic**: Beyond simple comprehensions and literals.
- **Explicit values**: Avoid magic numbers—name them.

**Rationale**: Readable tests fail clearly and localize faults.

### 5.1 No Test-Specific Code Paths in Production (Non-Negotiable)

- Production code must never contain branches, flags, or behavior that exists only to satisfy tests.
  - Examples of forbidden patterns: `if TESTING: ...`, checking `PYTEST_CURRENT_TEST`, special test-only parameters, alternate I/O paths only under tests.
- Tests must exercise the same public API and code paths used in production.
- Make code testable via proper seams instead:
  - Dependency injection with sensible production defaults (e.g., pass Path, clock, RNG, HTTP client).
  - Use pytest fixtures and mocks/monkeypatch for external boundaries (network, filesystem, time, env).
- Idempotency and determinism are product qualities, not test toggles. Implement them unconditionally where applicable.

### 6. Fixtures (Strict Usage)

- **Scope**: Prefer function-scoped; module/session only for expensive immutable data.
- **Location**: All shared setup in `tests/conftest.py`.
- **Purity**: Fixtures must be pure and return simple objects; avoid side effects.
- **I/O fixtures**: If writing to disk/network, must be session-scoped, opt-in, behind marker.

**Rationale**: Contained, fast, and reproducible tests.

### 7. Mocking and Fakes (Hard Limits)

- **Mock only external boundaries**: network (requests), filesystem, time, randomness, environment, subprocess.
- **Do not mock internal code**: Test behavior through public API.
- **Prefer fakes**: Over mocks where feasible (in-memory repositories, stub services).
- **No live HTTP**: All network calls must be mocked or faked.

**Rationale**: Stability and meaningful coverage.

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
- **No pragma comments**: ABSOLUTELY FORBIDDEN except for impossible-to-test code (if 0:, if **name** == **main**:)

**Rationale**: Complete test coverage ensures zero blind spots and maximum confidence in code quality.

### 12. Flaky Test Policy (IMMEDIATE ACTION REQUIRED)

**ABSOLUTE ZERO TOLERANCE**: Any flaky test is IMMEDIATELY removed from the suite. NO 24h grace period. NO xfail exceptions. NO second chances.

**Enforcement**: First flake = immediate deletion. Tests must be 100% deterministic and reliable.

**Rationale**: Even a single flaky test destroys developer confidence and wastes precious development time.

## Mutation Testing (Optional)

- We use Cosmic Ray for mutation testing; configuration lives in `pyproject.toml` under `[tool.cosmic-ray]`.
- Run manually when desired:

```bash
make quality-ext
```

- This initializes (if needed) and executes Cosmic Ray sessions at `out/cosmic-ray/session.sqlite`. The step is non-fatal and not part of CI or pre-commit by default.

### Running Tests

#### Local Development

```bash
# Fast check
make pytest-core PYARGS="-q"

# With markers (exclude slow/perf)
make pytest-core PYARGS='-m "not slow and not perf" -q'

# Full suite with coverage
make coverage
```

#### CI Commands

```bash
# Coverage check (CI)
make coverage
```

### Example Test Patterns

#### Parametrized Unit Test

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
        config = load_toml(config_path)
        assert config.model.block_size == block_size
    else:
        with pytest.raises(ValueError):
            load_toml(config_path)
```

#### Fixture with tmp_path

```python
import json
from pathlib import Path
import pytest

@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    return path
```

### Adding New Tests

Create files under `tests/unit/ml_playground/test_<module>.py`:

```python
from __future__ import annotations
from pathlib import Path
import pytest
from ml_playground.config import load_toml, TrainerConfig


def test_config_loading_handles_missing_file(tmp_path: Path) -> None:
    """Test that config loading raises appropriate error for missing files."""
    # Arrange
    missing_path = tmp_path / "missing.toml"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_toml(missing_path)
```

## Code Style Standards

### Automatic Formatting

Ruff automatically applies modern Python best practices:

- **Type annotations**: `typing.List` → `list`, `typing.Dict` → `dict`
- **Union syntax**: `Optional[str]` → `str | None`, `Union[A, B]` → `A | B`
- **Import organization**: Sorted and cleaned automatically
- **Code formatting**: Black-compatible formatting
- **Whitespace cleanup**: Trailing whitespace removal

### Development Guidelines

- **Strictly typed code**: Use explicit types and `pathlib.Path` for filesystem paths
- **Pure functions**: Favor pure functions for data preparation
- **Explicit device selection**: Make device selection explicit in code
- **Configuration and overrides**:
  - TOML is the primary source of truth; avoid ad-hoc CLI flags that mutate config.
  - Allowed exceptions (documented and tested):
    - Global CLI option `--exp-config PATH` to choose a specific experiment TOML; `experiments/default_config.toml` is merged first.
    - Environment JSON overrides: `ML_PLAYGROUND_TRAIN_OVERRIDES` and `ML_PLAYGROUND_SAMPLE_OVERRIDES`.
      These are deep-merged and then strictly re-validated; invalid env overrides are ignored.

## Tool Configuration Policy

**Single-source rule**: All tool configuration must live in `pyproject.toml` only.

**Prohibited**: Standalone config files (.ruff.toml, mypy.ini, pyrightconfig.json, pytest.ini, setup.cfg, etc.)

**Centralized sections**:

- `[tool.ruff]` for lint/format settings
- `[tool.mypy]` for type checker settings  
- `[tool.pyright]` for static analysis include/exclude
- `[tool.pytest.ini_options]` for pytest testpaths and options

## Architecture Notes

### Checkpointing and Resume

 Checkpoints are strictly rotated-only.

- Last checkpoints are saved with a timestamped suffix, e.g., `ckpt_last_XXXXXXXX.pt`.
- Best checkpoints are saved with a timestamped suffix and metric, e.g., `ckpt_best_XXXXXXXX_<metric>.pt`.
- On resume, checkpointed `model_args` override TOML for compatibility.
- To change model shapes: start with fresh `out_dir` or delete existing checkpoints.

### Dataset Notes

- Char-level Bundestag dataset autoseeds from bundled sample resource
- Replace with real data for non-trivial runs
- Use `device="cpu"` for tests and local CI

For import standards, see [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md).

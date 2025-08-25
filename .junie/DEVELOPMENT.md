# ml_playground Development Practices

Core development practices, quality standards, and workflow for ml_playground contributors.

## Quality Gates (Mandatory)

Run these commands before every commit:

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright
uv run mypy ml_playground
uv run pytest -n auto -W error --strict-markers --strict-config -v
```

**All four gates must pass.** Do not open a PR otherwise.

## Commit Standards

### Granular Commits Policy
- **One logical change per commit** (e.g., fix a test, adjust a config, refactor a function)
- **Keep commits under ~200 lines** unless unavoidable
- **Run quality gates before each commit**, not just before PR

### Practical Tips
- Stage hunks selectively: `git add -p`
- Separate mechanical formatting from semantic changes
- Split large features into reviewable increments
- Keep tests passing at every step

### Conventional Commit Format
**Required format**: `<type>(<scope>): <subject>`

**Types**: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert
**Scope**: module or area (e.g., trainer, config, guidelines, tests)
**Subject**: imperative, concise, lowercase (no trailing period)

**Examples**:
- `feat(trainer): write checkpoint sidecar JSON with decision inputs/outputs`
- `test(trainer): add tests for checkpoint sidecar schema and behavior`
- `chore(config): centralize tooling settings in pyproject.toml`
- `docs(guidelines): document pyproject-only config and granular commits`

## Testing Standards (ULTRA-STRICT POLICY - 100% SUCCESS REQUIRED)

### 1. Test Framework and Runner
- **Framework**: pytest only. Do not use unittest or nose.
- **Runner**: `uv run pytest`
- **Coverage**: `uv run coverage run -m pytest; uv run coverage report; uv run coverage xml`
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
- **No pragma comments**: ABSOLUTELY FORBIDDEN except for impossible-to-test code (if 0:, if __name__ == __main__:)

**Rationale**: Complete test coverage ensures zero blind spots and maximum confidence in code quality.

### 12. Flaky Test Policy (IMMEDIATE ACTION REQUIRED)
**ABSOLUTE ZERO TOLERANCE**: Any flaky test is IMMEDIATELY removed from the suite. NO 24h grace period. NO xfail exceptions. NO second chances.

**Enforcement**: First flake = immediate deletion. Tests must be 100% deterministic and reliable.

**Rationale**: Even a single flaky test destroys developer confidence and wastes precious development time.

### Running Tests

#### Local Development
```bash
# Fast check
uv run pytest -q

# With markers (exclude slow/perf)
uv run pytest -m "not slow and not perf" -q

# Full suite with coverage
uv run coverage run -m pytest -m "not perf"
uv run coverage report -m
```

#### CI Commands
```bash
# Coverage check (CI)
uv run coverage run -m pytest -m "not perf"
uv run coverage report -m
uv run coverage xml
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
- **TOML-only configuration**: Keep CLI free of config mutation logic

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
- Trainer writes `ckpt_last.pt` every eval; updates `ckpt_best.pt` on improvement
- On resume, checkpointed `model_args` override TOML for compatibility
- To change model shapes: start with fresh `out_dir` or delete existing checkpoints

### Dataset Notes
- Char-level Bundestag dataset autoseeds from bundled sample resource
- Replace with real data for non-trivial runs
- Use `device="cpu"` for tests and local CI

For import standards, see [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md).
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

## Testing Standards

### Test Organization
- **Structure**: `tests/unit/{module_name}/test_{module_name}.py`
- **One test file per module** for clear separation of concerns
- **100% test coverage** for stable modules with strict quality gates

### Test Quality Requirements
- All warnings treated as errors
- Comprehensive type hints in all test files
- Pure pytest style (no unittest legacy patterns)
- Strict markers and config validation

### Running Tests
```bash
# Full suite
uv run pytest -n auto -W error --strict-markers --strict-config -v

# Filtered tests
uv run pytest -n auto -W error --strict-markers --strict-config -v -k "config or data"
```

### Adding New Tests
Create files under `tests/unit/ml_playground/test_<name>.py`:

```python
from pathlib import Path
from ml_playground.config import load_toml, AppConfig

def test_example(tmp_path: Path) -> None:
    # Test implementation
    pass
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
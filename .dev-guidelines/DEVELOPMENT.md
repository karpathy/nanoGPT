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

### Runnable State Requirement (MANDATORY)

- Every commit MUST be in a runnable state when checked out.
- Runnable means:
  - `make quality` passes locally (same as pre-commit/CI gate).
  - No partially applied migrations or broken CLI entry points.
  - Documentation build (if modified) is not broken.
- Do not commit code that knowingly breaks the build with intent to "fix later". Split work into smaller, independently runnable commits.

### Branching Model (Feature Branches REQUIRED)

- All work MUST happen on short-lived feature branches. Do not commit directly to `main`.
- For naming conventions, linear history, and rebase policy, see `GIT_VERSIONING.md`.

### Conventional Commit Format

See `GIT_VERSIONING.md` for the required Conventional Commits format, examples, and usage notes.

## Testing Standards

For testing standards, practices, and examples, see [TESTING.md](TESTING.md).

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

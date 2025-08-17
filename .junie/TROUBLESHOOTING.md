# ml_playground Troubleshooting Guide

Common issues and solutions when working with ml_playground development environment.

## Environment Issues

### `uv venv` hangs or appears to stall

**Cause**: You likely invoked `uv venv` from within an already-activated virtual environment (including conda), which can interfere with environment creation.

**Solutions**:

**POSIX shells (Linux/macOS)**:
- Start a fresh shell, or run `deactivate` (or `conda deactivate`) first, then retry `uv venv`
- Run without inherited venv/conda variables: `env -u VIRTUAL_ENV -u CONDA_PREFIX uv venv`
- Select a specific interpreter: `uv venv --python $(command -v python3)` (or an absolute path like `/usr/bin/python3`)

**Windows/PowerShell**:
- Close the activated shell or run `deactivate` (or `conda deactivate`) and retry `uv venv`
- Optionally select interpreter: `uv venv --python py -3` (or a full path to python.exe)

**Corrupted environment**: If a previous `.venv` exists and is corrupted, remove it before recreating: `rm -rf .venv` (use with care)

## Import Issues

### Tests cannot import `ml_playground`

**Cause**: You are not running inside the project venv.

**Solution**: Run `uv venv` then `uv sync --all-groups` and execute commands with `uv run` from the project root.

### `uv run pytest` fails due to missing pytest

**Cause**: You did not sync dev tools.

**Solution**: Run `uv sync --all-groups` to install all dependency groups including dev tools.

## Platform-Specific Issues

### Torch wheels

**Issue**: Torch installation fails or uses wrong version.

**Solutions**:
- Use a supported Python version (see pyproject.toml)
- Run CPU-first configurations in tests
- For GPU support, ensure explicit device configuration in TOML files

## Development Workflow Issues

### Quality gates failing

**Common causes and solutions**:

**Ruff formatting issues**:
```bash
uv run ruff check --fix . && uv run ruff format .
```

**Type checking failures**:
```bash
uv run pyright  # for static analysis
uv run mypy ml_playground  # for type checking
```

**Test failures**:
```bash
# Run full test suite
uv run pytest -n auto -W error --strict-markers --strict-config -v

# Run specific tests for debugging
uv run pytest -n auto -W error --strict-markers --strict-config -v -k "config or data"
```

### Git commit issues

**Pre-commit checks failing**: Always run all quality gates before committing:
```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright
uv run mypy ml_playground
uv run pytest -n auto -W error --strict-markers --strict-config -v
```

**Large commits**: Break them into smaller, focused commits using `git add -p` for selective staging.

## Configuration Issues

### TOML configuration not loading

**Check**:
- File path is correct relative to project root
- TOML syntax is valid
- Required sections exist ([train] and/or [sample])

### Checkpoint resume issues

**Model shape mismatches**: 
- Start with fresh `out_dir` or delete existing `ckpt_last.pt`/`ckpt_best.pt`
- Checkpointed `model_args` override TOML for compatibility

### Device configuration

**GPU not being used**: 
- Explicitly set device in TOML configuration
- Verify GPU availability with appropriate drivers
- Use `device="cpu"` for development and testing

## Getting Help

If you encounter issues not covered here:

1. Check that you're following the setup steps in [SETUP.md](SETUP.md)
2. Verify your development environment matches [DEVELOPMENT.md](DEVELOPMENT.md) requirements
3. Review import standards in [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md) if import-related

For persistent issues, ensure you have a clean environment:
```bash
rm -rf .venv
uv venv --clear
uv sync --all-groups
```
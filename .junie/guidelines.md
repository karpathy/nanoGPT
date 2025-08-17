# ml_playground Developer Guidelines

**Audience**: Advanced contributors working exclusively on the `ml_playground` module. These rules are binding. Follow them exactly.

## Quick Start

Get up and running immediately:

```bash
uv venv --clear
uv sync --all-groups
uv run ruff check --fix . && uv run ruff format .
uv run pyright && uv run mypy ml_playground
uv run pytest -n auto -W error --strict-markers --strict-config -v
```

## Documentation Structure

This guideline system is organized into focused documents for easy navigation:

### 📋 [SETUP.md](SETUP.md) - Environment Setup
- Prerequisites and installation
- Virtual environment creation
- Basic workflow commands
- Configuration system overview
- Quick troubleshooting

### 🔧 [DEVELOPMENT.md](DEVELOPMENT.md) - Core Development Practices
- Quality gates and commit standards
- Testing workflow and organization
- Code style standards and tooling
- Architecture notes and best practices

### 📦 [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md) - Import Standards
- Strict import policies and rationale
- Canonical patterns and examples
- Review checklist and enforcement

### 🔍 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem Solving
- Common environment issues
- Import and dependency problems
- Platform-specific solutions
- Development workflow fixes

## Core Principles (Non-Negotiable)

**UV-Only Workflow**: Use UV for everything - virtualenv, dependency sync, running tools and tests. No pip, requirements.txt, or manual venv activation.

**Quality First**: All quality gates must pass before any commit/PR:
- Linting and formatting
- Static analysis 
- Type checking
- Tests with warnings as errors

**Granular Commits**: Make small, focused commits with conventional commit messages. Run quality gates before each commit, not just before PR.

**TOML Configuration**: All configuration via TOML files mapped to dataclasses. No CLI parameter overrides.

**Strict Typing**: Code is strictly typed with explicit types and pathlib.Path for filesystem paths.

## Essential Commands

**Environment Setup**:
```bash
uv venv --clear && uv sync --all-groups
```

**Quality Gates** (run before each commit):
```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright
uv run mypy ml_playground  
uv run pytest -n auto -W error --strict-markers --strict-config -v
```

**Runtime Entry Points**:
```bash
# Prepare datasets
uv run python -m ml_playground.cli prepare shakespeare
uv run python -m ml_playground.cli prepare bundestag_char

# Train from TOML
uv run python -m ml_playground.cli train ml_playground/configs/shakespeare_cpu.toml

# Sample from trained model
uv run python -m ml_playground.cli sample ml_playground/configs/shakespeare_cpu.toml

# End-to-end pipeline
uv run python -m ml_playground.cli loop bundestag_char ml_playground/configs/bundestag_char_cpu.toml
```

## Need Help?

- **Setup issues**: See [SETUP.md](SETUP.md) for installation and basic workflow
- **Development questions**: Check [DEVELOPMENT.md](DEVELOPMENT.md) for practices and standards  
- **Import problems**: Review [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md) for strict policies
- **Troubleshooting**: Consult [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

---

*This guideline system ensures consistent, high-quality development practices across the ml_playground module. Each document focuses on a specific aspect while maintaining coherent overall standards.*
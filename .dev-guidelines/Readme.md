---
trigger: always_on
---

# ml_playground Developer Guidelines

**Audience**: Advanced contributors working exclusively on the `ml_playground` module. These rules are binding. Follow them exactly.

## Quick Start

Get up and running immediately:

```bash
make setup
make verify
make quality   # ruff+format+pyright+mypy+pytest
```

## Documentation Structure

This guideline system is organized into focused documents for easy navigation:

### 📝 [DOCUMENTATION.md](DOCUMENTATION.md) - Documentation Guidelines

- Abstraction levels and required sections
- Annotated folder tree standard (with inline descriptions)
- Markdown style (markdownlint) and DRY docs policy
- Cross-referencing shared framework docs

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

### 🔀 [GIT_VERSIONING.md](GIT_VERSIONING.md) - Git Versioning & Workflow

- Feature branch model, naming conventions, and linear history/rebase policy
- Conventional Commits format with examples
- Runnable-commit verification gates and commit granularity

### 📦 [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md) - Import Standards

- Strict import policies and rationale
- Canonical patterns and examples
- Review checklist and enforcement

### 🔍 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem Solving

- Common environment issues
- Import and dependency problems
- Platform-specific solutions
- Development workflow fixes

**UV-Only Workflow**: Use UV for everything - virtualenv, dependency sync, running tools and tests. No pip, requirements.txt, or manual venv activation.

All documentation in this repo must adhere to [DOCUMENTATION.md](DOCUMENTATION.md).

## Core Principles (Non-Negotiable)

**Quality First**: All quality gates must pass before any commit/PR:

- Linting and formatting
- Static analysis
- Type checking
- Tests with warnings as errors

**Granular Commits**: Make small, focused commits with conventional commit messages. Run quality gates before each commit, not just before PR.

**Runnable Commits**: Every commit must be in a runnable state when checked out. Do not land commits that break `make quality`, CLI entry points, or docs builds. Never bypass verification (avoid `--no-verify`).

**Feature Branches Only**: All work happens on short‑lived feature branches; no direct commits to `main`. Use kebab-case names like `feat/<scope>-<short-desc>`, `fix/<scope>-<short-desc>`. Keep branches focused and prefer multiple small PRs.

**TOML Configuration**: TOML is the primary source of truth, mapped to dataclasses. No ad‑hoc CLI parameter overrides.

- Allowed exceptions (as implemented):
  - Global CLI option `--exp-config PATH` selects an alternative experiment TOML file (replaces the experiment’s `config.toml`). The global `experiments/default_config.toml` is still merged first under the experiment config.
  - Environment JSON overrides are supported, deep-merged, and strictly re-validated; invalid overrides are ignored to avoid breaking flows:
    - `ML_PLAYGROUND_TRAIN_OVERRIDES`
    - `ML_PLAYGROUND_SAMPLE_OVERRIDES`

**Strict Typing**: Code is strictly typed with explicit types and pathlib.Path for filesystem paths.

## Essential Commands

**Environment Setup**:

```bash
make setup
make verify
```

**Quality Gates** (run before each commit):

```bash
make quality
```

**Runtime Entry Points**:

```bash
# Prepare datasets
make prepare shakespeare
make prepare bundestag_char

# Train (select config path explicitly)
make train shakespeare CONFIG=ml_playground/configs/shakespeare_cpu.toml

# Sample from trained model
make sample shakespeare CONFIG=ml_playground/configs/shakespeare_cpu.toml

# End-to-end pipeline
make loop bundestag_char CONFIG=ml_playground/configs/bundestag_char_cpu.toml
```

## Need Help?

- **Setup issues**: See [SETUP.md](SETUP.md) for installation and basic workflow
- **Development questions**: Check [DEVELOPMENT.md](DEVELOPMENT.md) for practices and standards  
- **Import problems**: Review [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md) for strict policies
- **Troubleshooting**: Consult [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

## 🧪 Testing Docs

- Unit tests: see `tests/unit/README.md` (fast, isolated, pure-Python; no external TOML)
- Integration tests: see `tests/integration/README.md` (compose real modules via Python APIs)
- End-to-end (E2E) tests: see `tests/e2e/README.md` (CLI wiring, config merge, logging)
  - Use explicit tiny defaults for E2E CLI runs:
    `--exp-config tests/e2e/ml_playground/experiments/test_default_config.toml`

---

*This guideline system ensures consistent, high-quality development practices across the ml_playground module. Each document focuses on a specific aspect while maintaining coherent overall standards.*

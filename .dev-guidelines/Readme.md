---
trigger: always_on
description: Entry point for ml_playground developer guidelines and core policies
---

# ml_playground Developer Guidelines

**Audience**: Advanced contributors working exclusively on the `ml_playground` module. These rules are binding. Follow
them exactly.

<details>
<summary>Related documentation</summary>

- [Development Practices](./DEVELOPMENT.md) – Core development practices, quality standards, and workflow for ml_playground contributors.
- [Documentation Guidelines](./DOCUMENTATION.md) – Unified standards for documentation structure, abstraction levels, and formatting.
- [Setup Guide](./SETUP.md) – Quick start instructions for preparing the ml_playground development environment.

</details>

## Table of Contents

- [Quick Start](#quick-start)
- [Documentation Structure](#documentation-structure)
- [Core Principles (Non-Negotiable)](#core-principles-non-negotiable)
- [Essential Commands](#essential-commands)
- [Need Help?](#need-help)
- [🧪 Testing Docs](#%f0%9f%a7%aa-testing-docs)

## Quick Start

Get up and running immediately:

```bash
uvx --from . env-tasks setup
uvx --from . env-tasks verify
uvx --from . ci-tasks quality   # ruff + format + pyright + mypy + pytest
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

**UV-Only Workflow**: Use UV for everything - virtualenv, dependency sync, running tools and tests. No pip,
requirements.txt, or manual venv activation.

All documentation in this repo must adhere to [DOCUMENTATION.md](DOCUMENTATION.md).

## Core Principles (Non-Negotiable)

- **Quality gates stay green**: Always run `uvx --from . ci-tasks quality` before committing. Deep details live in [`DEVELOPMENT.md`](./DEVELOPMENT.md#quality-gates-mandatory).
- **TDD and commit pairing**: Follow the commit policy in [`DEVELOPMENT.md`](./DEVELOPMENT.md#commit-standards) and the strict TDD workflow in [`TESTING.md`](./TESTING.md#test-driven-development-required).
- **Feature branches + conventional commits**: Required naming and history rules are documented in [`GIT_VERSIONING.md`](./GIT_VERSIONING.md).
- **Configuration via TOML**: Treat configuration as single-source per [`SETUP.md`](./SETUP.md#configuration-system) and [`DEVELOPMENT.md`](./DEVELOPMENT.md#guiding-principles).
- **Strict typing + UV tooling**: See [`DEVELOPMENT.md`](./DEVELOPMENT.md#guiding-principles) for typing, tooling, and reuse expectations.

## Essential Commands

**Environment Setup**: Follow [`SETUP.md#environment-setup`](SETUP.md#environment-setup) to create and verify the UV environment.

**Quality Gates**: Run `uvx --from . ci-tasks quality` before each commit; see [`DEVELOPMENT.md#quality-gates-mandatory`](DEVELOPMENT.md#quality-gates-mandatory) for full rationale.

**Runtime Entry Points**: Use the Typer CLI commands documented in [`SETUP.md#basic-workflow-commands`](SETUP.md#basic-workflow-commands) and the flows outlined in [`ARCHITECTURE.md#runtime-entry-points`](ARCHITECTURE.md#runtime-entry-points).

## Need Help?

- **Setup issues**: See [SETUP.md](SETUP.md) for installation and basic workflow
- **Development questions**: Check [DEVELOPMENT.md](DEVELOPMENT.md) for practices, standards, and architecture pointers
- **Import problems**: Review [IMPORT_GUIDELINES.md](IMPORT_GUIDELINES.md) for strict policies
- **Troubleshooting**: Consult [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

## 🧪 Testing Docs

- Unit tests: see `tests/unit/README.md` (fast, isolated, pure-Python; no external TOML)
- Integration tests: see `tests/integration/README.md` (compose real modules via Python APIs)
- End-to-end (E2E) tests: see `tests/e2e/README.md` (CLI wiring, config merge, logging)
  - Use explicit tiny defaults for E2E CLI runs:
    `--exp-config tests/e2e/ml_playground/experiments/test_default_config.toml`

______________________________________________________________________

*This guideline system ensures consistent, high-quality development practices across the ml_playground module. Each
document focuses on a specific aspect while maintaining coherent overall standards.*

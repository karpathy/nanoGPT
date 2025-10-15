---
trigger: manual
description: Implementation notes for GitHub automation, workflows, and operational tooling
---

# GitHub Automation Overview

Implementation-specific reference for the `.github/` directory, detailing how our CI workflows are structured and operated on GitHub Actions.

<details>
<summary>Related documentation</summary>

- [Continuous Integration Guidelines](../.dev-guidelines/CI.md) – Platform-agnostic CI policies and maintenance checklist.
- [Development Practices](../.dev-guidelines/DEVELOPMENT.md) – Core quality gates, tooling, and workflow expectations.
- [Testing Standards](../.dev-guidelines/TESTING.md) – Required TDD workflow and test organization.

</details>

## Table of Contents

- [Directory Overview](#directory-overview)
- [Workflow Summary](#workflow-summary)
- [Caching Implementation](#caching-implementation)
- [Operational Commands](#operational-commands)
- [Maintenance Notes](#maintenance-notes)

## Directory Overview

```bash
.github/
├── README.md               # GitHub automation reference (this file)
├── workflows/              # GitHub Actions workflow definitions
│   ├── quality.yml         # Fast gate covering linting, typing, and test suite
│   └── mutation-suite.yml  # Scheduled/manual mutation testing pipeline
├── copilot-instructions.md # Supplemental settings for GitHub Copilot
├── ARCHITECTURE.md         # Mirrors core architecture guidelines for external links
├── DEVELOPMENT.md          # Mirrors development practices for external links
├── DOCUMENTATION.md        # Mirrors documentation guidelines for external links
├── GIT_VERSIONING.md       # Mirrors versioning rules for external links
├── IMPORT_GUIDELINES.md    # Mirrors import policy for external links
├── REQUIREMENTS.md         # Mirrors dependency policy for external links
├── SETUP.md                # Mirrors environment setup guide for external links
└── TESTING.md              # Mirrors testing standards for external links
```

## Workflow Summary

- **`quality.yml`**
  - **Purpose**: Mandatory gate executed on every push/PR to enforce linting, formatting, typing, and tiered pytest suites via `uv run ci-tasks quality`.
  - **Triggers**: `push`/`pull_request` to active branches.
  - **Timeout**: 30 minutes (default runner limit; job typically completes in ~2 minutes when caches hit).
  - **Caching**: Restores `.venv`, pre-commit, and ruff caches; relies on `astral-sh/setup-uv@v6` for wheel caching with pruning enabled.
  - **Manual use**: `gh workflow run quality.yml --ref <branch>`.
- **`mutation-suite.yml`**
  - **Purpose**: Executes the mutation testing suite (`uv run ci-tasks mutation run`) and captures reports.
  - **Triggers**: Weekly cron (`0 1 * * 1`) and manual `workflow_dispatch` for investigative runs.
  - **Timeout**: Explicit 180-minute limit to cap long-running mutation jobs.
  - **Prerequisites**: Installs `python3-dev`, `build-essential`, `libffi-dev`, and `gfortran` before dependency sync.
  - **Caching**: Shares the same `.venv` cache key as `quality.yml`; also prunes the uv wheel cache post-run.
  - **Artifacts**: Uploads `mutation-report.txt` and `.cache/cosmic-ray/session.sqlite` on completion.

See [`workflows/README.md`](workflows/README.md) for per-file implementation details, command snippets, and change history highlights.

## Caching Implementation

- **Virtual environment (`.venv`)**: Managed via `actions/cache@v4` keyed on `${{ runner.os }}-${{ python-version }}-uv-venv-${{ hashFiles('uv.lock') }}` with restore keys for broader reuse.
- **Wheel cache (`.cache/uv`)**: Handled by `astral-sh/setup-uv@v6` using `enable-cache: true`, `cache-local-path: .cache/uv`, and `prune-cache: true`.
- **Tool caches**: Pre-commit (`.cache/pre-commit`) and ruff (`.cache/ruff`) are cached separately to avoid invalidating the `.venv` when configs change.
- **Pruning**: Both workflows run `uv cache prune --ci` after completing tasks that modify `.cache/uv` to keep uploads small without touching the restored `.venv`.

## Operational Commands

- **Trigger a workflow**: `gh workflow run <workflow>.yml --ref <branch>`.
- **Monitor a run**: `gh run watch <run-id>` for full streaming logs.
- **Sample progress**: `timeout 180 gh run watch <run-id>` to observe the first three minutes and exit (workflow continues server-side).
- **Cancel a run**: `gh run cancel <run-id>` to free runner capacity.
- **List recent runs**: `gh run list --workflow <workflow>.yml --limit 5`.

## Maintenance Notes

- Reflect any workflow changes (new jobs, triggers, cache keys) in this README and in `../.dev-guidelines/CI.md` during the same pull request.
- When bumping action versions or adding system dependencies, document the change under the relevant workflow section.
- Review scheduled workflows quarterly to validate cron cadence, credential freshness, and runtime budgets.
- Keep this README focused on GitHub-specific implementation; platform-agnostic principles belong in the central CI guidelines.

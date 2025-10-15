---
trigger: manual
description: File-level reference for GitHub Actions workflow definitions
---

# GitHub Actions Workflows

This document captures implementation details, schedules, and operational tips for each workflow in `.github/workflows/`.

<details>
<summary>Related documentation</summary>

- [GitHub Automation Overview](../README.md) – Directory-level overview and operational commands.
- [Continuous Integration Guidelines](../../.dev-guidelines/CI.md) – Platform-agnostic CI policies.
- [Development Practices](../../.dev-guidelines/DEVELOPMENT.md) – Quality gates, tooling, and workflow expectations.

</details>

## Table of Contents

- [quality.yml](#qualityyml)
- [mutation-suite.yml](#mutation-suiteyml)
- [Change Log](#change-log)

## quality.yml

- **Purpose**: Fast gate validating linting, formatting, typing, and pytest tiers via `uv run ci-tasks quality`.
- **Triggers**: `push` and `pull_request` for active branches.
- **Timeout**: Default runner limit (job completes in ~2 minutes with warm caches).
- **Cache usage**:
  - `.venv` restored via `actions/cache@v4` keyed on `${{ runner.os }}-${{ python-version }}-uv-venv-${{ hashFiles('uv.lock') }}`.
  - `.cache/pre-commit` and `.cache/ruff` cached separately to avoid invalidating `.venv`.
  - `.cache/uv` handled by `astral-sh/setup-uv@v6` with `enable-cache: true`, `cache-local-path: .cache/uv`, and `prune-cache: true`.
- **Key steps**:
  - `uv sync --frozen --group dev` guarded by a cache-hit check.
  - `uv run ci-tasks quality` executes pre-commit, linting, typing, and pytest suites.
  - `uv cache prune --ci` runs post-tests to shrink the uv wheel cache.
  - Coverage badge verification fails the job if docs/assets/coverage SVGs drift.
- **Manual run**: `gh workflow run quality.yml --ref <branch>`.

## mutation-suite.yml

- **Purpose**: Runs mutation testing (`uv run ci-tasks mutation run`) and captures reports for analysis.
- **Triggers**: Weekly cron (`0 1 * * 1`) plus `workflow_dispatch` for ad hoc runs.
- **Timeout**: 180 minutes.
- **Prerequisites**: Installs `python3-dev`, `build-essential`, `libffi-dev`, and `gfortran` before dependency sync to support packages such as `pycocotools`.
- **Cache usage**:
  - `.venv` cache identical to `quality.yml`, enabling shared reuse.
  - Wheel cache managed by `setup-uv` with pruning to keep uploads small.
- **Key steps**:
  - Conditional `uv run env-tasks sync --group dev` when `.venv` cache misses.
  - `uv run ci-tasks mutation run` executes the mutation suite.
  - `uv run python tools/mutation_report.py --config pyproject.toml | tee mutation-report.txt` captures summary stats.
  - Artifacts (`mutation-report.txt`, `.cache/cosmic-ray/session.sqlite`) uploaded even on failure.
- **Operational tips**:
  - Use `timeout 180 gh run watch <run-id>` to sample early progress.
  - Cancel superseded runs with `gh run cancel <run-id>` to release the runner.

## Change Log

| Date (UTC) | Change                                                                                                           | Notes                           |
| ---------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| 2025-10-15 | Aligned caching strategy, added build prerequisites for mutation suite, and upgraded to `astral-sh/setup-uv@v6`. | Documented in PR `#75`.         |
| 2025-10-15 | Added manual dispatch guidelines, timeout sampling approach, and artifact expectations.                          | Initial version of this README. |

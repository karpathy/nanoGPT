---
trigger: always_on
description: Branching model, Conventional Commit rules, and linear history practices
---

# Git Versioning & Workflow

Canonical rules for branches, commit messages, and history management. Keep commits runnable and history linear to
maintain velocity and reliability.

<details>
<summary>Related documentation</summary>

- [Developer Guidelines Index](./Readme.md) – Entry point for ml_playground principles and quick-start commands.
- [Development Practices](./DEVELOPMENT.md) – Quality gates, commit standards, and runnable-state expectations.

</details>

## Table of Contents

- [Branching Model (Feature Branches REQUIRED)](#branching-model-feature-branches-required)
- [Conventional Commits](#conventional-commits)
- [Linear History & Rebasing](#linear-history--rebasing)
- [Verification Gates & Runnable Commits](#verification-gates--runnable-commits)
- [Commit Granularity (Reference)](#commit-granularity-reference)

## Branching Model (Feature Branches REQUIRED)

- All work happens on short‑lived feature branches; no direct commits to `main`.
- Naming (kebab-case):
  - `feat/<scope>-<short-desc>`
  - `fix/<scope>-<short-desc>`
  - `chore/<scope>-<short-desc>`
  - `docs/<scope>-<short-desc>`
- Keep branches focused; prefer multiple small PRs over a single large PR.

## Conventional Commits

Required format: `<type>(<scope>): <subject>`

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
- Scope: module or area (e.g., `trainer`, `config`, `guidelines`, `tests`)
- Subject: imperative, concise, lowercase, no trailing period

Examples:

- `feat(trainer): write checkpoint sidecar JSON with decision inputs/outputs`
- `test(trainer): add tests for checkpoint sidecar schema and behavior`
- `chore(config): centralize tooling settings in pyproject.toml`

### Meaningful subjects required

- Subject lines must describe *what* changed and, when useful, *why*.
- Avoid vague rollups such as "update tests" or "misc fixes".
- Mention the affected component so reviewers can infer impact quickly.

Good examples:

- `test(cli): add cuda device seeding property`
- `docs(workflow): clarify optional ci-tasks quality usage`
- `fix(loader): guard against missing override file`

Poor examples (do not use):

- `expand cli and data pipeline coverage`
- `more tweaks`
- `update stuff`

## Linear History & Rebasing

- Maintain a linear history for your work: rebase on top of the target branch; avoid merge commits; fast‑forward only.
- Squash when appropriate to keep commits meaningful and reviewable.
- Resolve conflicts locally and re-run gates before pushing.

## Verification Gates & Runnable Commits

- Every commit must be runnable when checked out.
- Gates to pass locally (same as pre-commit/CI): `uv run ci-tasks quality`.
- Do not bypass verification (avoid `--no-verify`).
- See `.dev-guidelines/DEVELOPMENT.md` → “Runnable State Requirement (MANDATORY)” for details.

## Commit Granularity (Reference)

- One logical change per commit; keep commits small.
- Pair functional changes with tests in the same commit (TDD workflow).
- Documentation-only and mechanical changes may be committed separately.

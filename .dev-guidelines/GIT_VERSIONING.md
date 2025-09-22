---
trigger: always_on
description: Git branching, versioning, and commit policy
---

# Git Versioning & Workflow

Canonical rules for branches, commit messages, and history management. Keep commits runnable and history linear to maintain velocity and reliability.

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

## Linear History & Rebasing

- Maintain a linear history for your work: rebase on top of the target branch; avoid merge commits; fast‑forward only.
- Squash when appropriate to keep commits meaningful and reviewable.
- Resolve conflicts locally and re-run gates before pushing.

## Verification Gates & Runnable Commits

- Every commit must be runnable when checked out.
- Gates to pass locally (same as pre-commit/CI): `make quality`.
- Do not bypass verification (avoid `--no-verify`).
- See `.dev-guidelines/DEVELOPMENT.md` → “Runnable State Requirement (MANDATORY)” for details.

## Commit Granularity (Reference)

- One logical change per commit; keep commits small.
- Pair functional changes with tests in the same commit (TDD workflow).
- Documentation-only and mechanical changes may be committed separately.

# DEVELOPMENT.md — Core Development Practices

Audience: Contributors working on the ml_playground module.

This document summarizes the core practices used in this repository with special emphasis on testing guidelines.

## Quality Gates (pre-commit)
Run these before every commit/PR:

```bash
uv venv --clear && uv sync --all-groups   # first time or when deps change
uv run ruff check --fix . && uv run ruff format .
uv run pyright
uv run mypy ml_playground
uv run pytest -n auto -W error --strict-markers --strict-config -v
```

## Typing & Config
- Strict typing everywhere; prefer explicit types.
- Use pathlib.Path for file paths.
- All runtime configuration via TOML mapped to dataclasses (see ml_playground/config.py). No CLI overrides.

## Testing Guidelines (Non‑Negotiable)
- Production code never has special treatment for tests.
  - Do not add branches, flags, or environment checks in production code to accommodate tests (e.g., `if TESTING: ...`, `if 'PYTEST_CURRENT_TEST' in os.environ`, special test-only parameters, or alternate code paths “just for tests”).
  - Do not write to alternate locations or use alternate I/O semantics only when running tests.
- Tests must exercise public (or well-defined internal) APIs as-is.
- Prefer test doubles (mocks, stubs), fixtures, and dependency injection to make code testable:
  - Inject collaborators (e.g., file system paths, I/O functions, HTTP clients) as parameters when appropriate, with sensible defaults for production use.
  - Use pytest fixtures and monkeypatch/mocks to replace external effects in tests.
- Determinism: tests must be deterministic and independent (no shared state across tests; use tmp_path for filesystem writes).
- Idempotency: preparation/IO utilities should be safe to call multiple times where it makes sense (e.g., skip when outputs are up‑to‑date) — this is a product requirement, not a test-only behavior.
- Isolation: no network calls in unit tests unless explicitly mocked; integration tests may opt-in with markers.

### Anti‑patterns (avoid)
- Test-only code paths in production source files.
- Hidden switches like environment variables, magic filenames, or global flags used only in tests.
- Overfitting production code to the specifics of a test’s mocking strategy.

### Recommended patterns
- Keep side effects localized and behind small, injectable seams.
- Split pure logic from I/O:
  - Pure functions are easy to test without special hooks.
  - Thin I/O wrappers can be mocked or fed temporary paths/streams in tests.
- Use realistic but small sample data under experiments/<name>/datasets when appropriate.

## Code Style
- Follow ruff’s rules (see pyproject.toml).
- Keep modules cohesive and small; use clear names and docstrings.
- Avoid dynamic imports except where necessary for optional integrations, and guard them cleanly.

## Commits
- Small, focused commits with conventional messages.
- Run quality gates locally before committing.

## Git Workflow: Linear history (Rebase, no merge commits for own branches)
- Do not use `git merge` to integrate upstream into your own feature branches. Always rebase on top of the latest upstream branch (e.g., `master`).
- Only fast-forward updates to protected branches. If a merge would create a merge commit, rebase first and then fast-forward.
- Recommended Git configuration:
  ```bash
  git config --global pull.rebase true
  git config --global rebase.autostash true
  git config --global merge.ff only
  git config --global branch.autosetuprebase always
  ```
- Typical feature flow:
  ```bash
  # start a feature
  git checkout -b feat/my-change
  # ...work, commit

  # keep up to date without merge commits
  git fetch origin
  git rebase origin/master  # resolve conflicts if any

  # run quality gates before publishing
  uv run ruff check --fix . && uv run ruff format .
  uv run pyright && uv run mypy ml_playground
  uv run pytest -n auto -W error --strict-markers --strict-config -v

  # publish rebased branch safely
  git push --force-with-lease -u origin HEAD
  ```
- Integrating multiple feature branches: rebase them onto each other in the intended order, then fast-forward the target branch to the final tip. Avoid merge commits.
- Prohibited: Merge commits when integrating your own branches (use rebase+ff only). For third‑party PRs, maintainers may choose an appropriate strategy but should prefer linear fast‑forward where possible.

## CI/Review Checklist
- [ ] Lint/format clean
- [ ] pyright clean
- [ ] mypy clean for ml_playground
- [ ] Tests pass with warnings as errors
- [ ] No test-specific branches in production code
- [ ] TOML-only configuration respected

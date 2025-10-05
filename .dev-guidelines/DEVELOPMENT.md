______________________________________________________________________

## trigger: always_on description: globs:

# ml_playground Development Practices

Core development practices, quality standards, and workflow for ml_playground contributors.

## Quality Gates (Mandatory)

For detailed information about the centralized framework utilities, see [Framework Utilities
Documentation](../docs/framework_utilities.md).

The pre-commit hook and CI both execute `make quality`. That target expands to ruff (lint + format),
pyright, mypy (scoped to `ml_playground`), and the targeted pytest suite. The command now runs
pre-commit with `--jobs $(PRE_COMMIT_JOBS)`, defaulting to `min(os.cpu_count(), 8)`; override via
`make quality PRE_COMMIT_JOBS=4` when you want to cap parallelism (e.g., inside containers).

During active development you may run narrower commands to iterate quickly, for example:

```bash
uv run pytest tests/property/cli/test_cli_property.py
uv run ruff check path/to/file.py
```

When you want the convenience wrappers (parallelized pytest, cache-aware collection, etc.), use the
Make targets directly. During tight iteration, reach for `make quality-fast` to run only the
formatting hooks (`ruff`, `ruff-format`, `mdformat`) with the same parallelism flags before kicking
off the heavier type and test gates:

```bash
make coverage-report      # run property + unit suites with coverage output
make tests-property       # property-based suites only
make tests-unit           # deterministic unit suites
make quality-fast         # lint/format the whole tree quickly
make lint                 # ruff lint+format without type checking
```

## Commit Standards

### Granular Commits Policy

- **One logical change per commit** (e.g., fix a test, adjust a config, refactor a function)
- **Keep commits under ~200 lines** unless unavoidable
- **Ensure quality gates pass before the commit is recorded** (the pre-commit hook enforces this automatically)
- **Pairing rule (REQUIRED)**: Each functional or behavioral change MUST include its tests in the same commit (unit
  and/or integration). Creating new files (untracked) is expected when adding tests—stage them together with the
  production change.
- **Granularity**: Prefer a short sequence of TDD commits to a single large commit. Each step should keep the suite
  green.

### Practical Tips

- Stage hunks selectively: `git add -p`
- Separate mechanical formatting from semantic changes
- Split large features into reviewable increments
- Keep tests passing at every step
- Pair production and test changes: when adding/refactoring code, include the minimal tests that specify the behavior in
  the same commit.
- Acceptable exceptions: documentation only commits, pure test refactors (no behavior change), or mechanical formatting.
  For everything else, pair code+tests.

### Runnable State Requirement (MANDATORY)

- Every commit MUST be in a runnable state when checked out.
- Runnable means:
  - Pre-commit (and therefore `make quality`) passes when the commit is created. Do not bypass hooks or suppress failures.
  - No partially applied migrations or broken CLI entry points.
  - Documentation build (if modified) is not broken.
- Do not commit code that knowingly breaks the build with intent to "fix later". Split work into smaller, independently
  runnable commits.

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
    - Global CLI option `--exp-config PATH` to choose a specific experiment TOML; `experiments/default_config.toml` is
      merged first.
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

## Dev Tooling Quick Reference

Use these commands from the repository root (or specify `--repo`/`--cwd` when required). They are
optimized for non-interactive or copy-paste workflows.

### uv / uvx

- **One-off commands**:

  ```bash
  uvx python -m http.server 8000
  uvx ruff check ml_playground/
  ```

- **Locked sync (no drift)**:

  ```bash
  uv sync --locked
  ```

- **Regenerate lockfile with explicit upgrade**:

  ```bash
  uv lock --upgrade
  ```

### ripgrep (`rg`)

- **Search code (smart case)**:

  ```bash
  rg "CheckpointManager" ml_playground/
  ```

- **Search tests only**:

  ```bash
  rg --glob 'tests/**' "make quality"
  ```

### GitHub CLI (`gh`)

- **Create draft PR (copy existing description file)**:

  ```bash
  gh pr create --draft --base master --title "docs: add dev tooling quick reference" --body-file .ldres/pr-body.md
  ```

- **Watch PR status (non-interactive)**:

  ```bash
  gh pr checks --watch
  ```

- **Open PR in browser**:

  ```bash
  gh pr view --web
  ```

#### CI inspection shortcuts

- **List recent runs for a branch**:

  ```bash
  gh run list --branch chore/python-version-alignment --limit 5 --json databaseId,headSha,status,conclusion,url
  ```

- **Surface failing runs only**:

  ```bash
  gh run list --branch chore/python-version-alignment --status failure --json databaseId,url
  ```

- **Inspect job-level status for a run**:

  ```bash
  gh api repos/mehrmorgen/nanoGPT/actions/runs/18234316534/jobs --jq '.jobs[] | {name,status,conclusion,log_url}'
  ```

#### PR management helpers

- **One-shot PR status**:

  ```bash
  gh pr checks 41
  ```

- **Merge with rebase and clean branches**:

  ```bash
  gh pr merge 41 --rebase --delete-branch
  ```

- **Find PRs for the current branch**:

  ```bash
  gh pr list --head docs/dev-tooling
  ```

### `fzf` helpers

- **Interactive file picker feeding `rg`**:

  ```bash
  rg --files | fzf | xargs open
  ```

- **Pick Make targets**:

  ```bash
  awk -F: '/^[a-zA-Z0-9_-]+:/ {print $1}' Makefile | sort -u | fzf | xargs -r make
  ```

### Non-interactive status & logs

- **Tail quality logs (CI artifact)**:

  ```bash
  gh run download --name quality --dir /tmp/mlp-quality && tail -n 200 /tmp/mlp-quality/*/logs.txt
  ```

- **Follow local development server**:

  ```bash
  uvx python scripts/run_dev_server.py 2>&1 | tee /tmp/mlp-dev.log
  ```

- **Check git status without pager**:

  ```bash
  git -c pager.status=cat status --short --branch
  ```

- **Audit branch tracking info**:

  ```bash
  git branch -vv
  ```

- **List merged branches (safe to delete)**:

  ```bash
  git branch --merged
  ```

- **Prune local branches with no remote**:

  ```bash
  git remote prune origin
  ```

- **See commits not yet on master**:

  ```bash
  git rev-list refactor/p11-di-samplerconfig ^master
  ```

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

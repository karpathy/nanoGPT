# ml-playground: strict, typed, UV-only training/sampling module

![Line Coverage](docs/assets/coverage-lines.svg)
![Branch Coverage](docs/assets/coverage-branches.svg)

This module provides a single, one-way interface to prepare data, train, and sample.
It is CPU/MPS-friendly, strictly typed, and uses TOML configs.

- Developer Guidelines: see `.dev-guidelines/Readme.md` for setup, development workflow, and policies.
- Tools index: see `tools/README.md` for helper scripts and usage.

## Documentation abstraction policy

- Top-level docs are high-level and describe the why and the overall layout.
- Each subfolder contains its own `Readme.md` with a focused scope and a folder tree.
- The deeper you go in the directory tree, the lower the level of abstraction and the more operational details you’ll find.

## Repository structure (high-level)

```bash
.
├── src/
│   └── ml_playground/         # core module (configs, experiments, runtime code)
│       ├── analysis/          # analysis tools (e.g., LIT integration)
│       ├── datasets/          # optional package; experiments can run without it
│       ├── experiments/       # self-contained experiments (mid-level docs)
│       └── configs/           # example configs referenced by docs/CLIs
├── tests/                     # test suite (see per-folder README for scope)
│   ├── unit/                  # low-level API tests
│   ├── integration/           # multi-module tests via Python APIs
│   ├── e2e/                   # CLI-level smoke tests
│   └── acceptance/            # higher-level behaviors and policies
├── tools/                     # developer tooling CLIs (see tools/README.md)
│   ├── ci_tasks.py            # uv-backed quality, coverage, mutation flows
│   ├── env_tasks.py           # uv-backed environment helpers
│   ├── lint_tasks.py          # uv-backed lint bundles
│   ├── lit_tasks.py           # uv-backed LIT helpers
│   └── test_tasks.py          # uv-backed pytest orchestration
├── docs/                      # supplementary docs (framework utilities, LIT, etc.)
├── pyproject.toml             # strict typing/linting/testing configuration
└── README.md                  # this file (top-level, high abstraction)

## Policy

- Use the uv-backed Typer CLIs for all workflows (env setup, quality, tests, runtime):
  - `uv run cli <command>` for experiment pipelines (`prepare`, `train`, `sample`).
  - `uv run env-tasks <command>` for environment setup, cache cleanup, TensorBoard, and AI-guideline helpers.
  - `uv run lint-tasks <command>` for lint/format bundles when you need faster feedback.
  - `uv run test-tasks <command>` for pytest suites.
  - `uv run ci-tasks <command>` for end-to-end quality gates, coverage generation, and mutation workflows.
- The project uses a `src/` layout. The uv CLIs automatically expose `src/` so `ml_playground` is importable without editable installs.
- Quality tooling is mandatory before commit (ruff, mypy, pyright), and tests must pass.
- Linear history for own work: rebase your branches and avoid merge commits; fast-forward only. See `.dev-guidelines/Readme.md` for developer policies.
- Test-Driven Development (TDD) is required for functional changes: write a failing test, implement minimal code to pass, then refactor.
- Granular commits are required. Each functional/behavioral change MUST pair its production code with the corresponding tests in the same commit (unit/integration). Exceptions: documentation-only, test-only refactors, and mechanical formatting.
- Review comment triage: use `uv run python tools/review.py list --pr <number> --unreplied --unresolved` to spot pending feedback; map comment URLs/IDs in `replies.json` for `bulk-reply`, and list comment IDs in `delete.json` for `uv run python tools/review.py delete --pr <number> --comments delete.json`.

Setup and Developer Workflow

- See `.dev-guidelines/Readme.md` for environment setup, development practices, and testing policies (entry point to all developer guidelines).

Datasets

- Shakespeare (GPT-2 BPE; prepared via internal ml_playground.experiments.shakespeare)
- Bundestag (char-level; prepared via internal ml_playground.experiments.bundestag_char; requires a user-provided text at src/ml_playground/experiments/bundestag_char/datasets/input.txt)
- Bundestag (tiktoken BPE; prepared via internal ml_playground.experiments.bundestag_tiktoken)

Workflows (high-level)

- Prepare/train/sample workflows are driven by the built-in Typer CLI: `uv run cli <command>`. For exact commands, refer to each experiment's `Readme.md` and `.dev-guidelines/Readme.md`.
- Universal meta policy: the data directory must contain a `meta.pkl` file used by training and sampling. The `prepare` step is responsible for writing `meta.pkl`.

Notes

- Configuration is defined via TOML dataclasses under `src/ml_playground/configuration/`.
- CPU/MPS are first-class. CUDA may be selected in TOML if available.
- Checkpoint behavior and policies are described in `.dev-guidelines/Readme.md`.
- For framework utilities, see [Framework Utilities Documentation](docs/framework_utilities.md).
- CLI validations: train and sample commands now fail fast if `meta.pkl` is missing.

Mutation testing

- See `.dev-guidelines/Readme.md` for how to run optional mutation testing (Cosmic Ray).

TensorBoard (auto-enabled)

- Training logs to TensorBoard. See `.dev-guidelines/Readme.md` for commands.

GGUF export (vendor approach)

- See `tools/llama_cpp/README.md` for the exact steps.

Testing

- See `.dev-guidelines/Readme.md` for testing standards and gates.
- See `tests/*/README.md` for folder-specific scope and patterns.
```

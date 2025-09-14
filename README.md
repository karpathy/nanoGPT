# ml-playground: strict, typed, UV-only training/sampling module

This module provides a single, one-way interface to prepare data, train, and sample.
It is CPU/MPS-friendly, strictly typed, and uses TOML configs.

## Documentation abstraction policy

- Top-level docs are high-level and describe the why and the overall layout.
- Each subfolder contains its own `Readme.md` with a focused scope and a folder tree.
- The deeper you go in the directory tree, the lower the level of abstraction and the more operational details you’ll find.

## Repository structure (high-level)

```bash
.
├── ml_playground/             # core module (configs, experiments, runtime code)
│   ├── analysis/              # analysis tools (e.g., LIT integration)
│   ├── datasets/              # optional package; experiments can run without it
│   ├── experiments/           # self-contained experiments (mid-level docs)
│   └── configs/               # example configs referenced by README/Makefile
├── tests/                     # test suite (see per-folder README for scope)
│   ├── unit/                  # low-level API tests
│   ├── integration/           # multi-module tests via Python APIs
│   ├── e2e/                   # CLI-level smoke tests
│   └── acceptance/            # higher-level behaviors and policies
├── tools/                     # developer tools and vendor integrations
│   └── llama_cpp/             # GGUF conversion helper (vendored instructions)
├── docs/                      # supplementary docs (framework utilities, LIT, etc.)
├── lit_nlp/                   # optional LIT integration
├── Makefile                   # entrypoints for setup, quality gates, runtime
├── pyproject.toml             # strict typing/linting/testing configuration
└── README.md                  # this file (top-level, high abstraction)

## Policy

- Use Make targets for all workflows (env setup, quality, tests, runtime). Under the hood, they run via uv.
- Never set PYTHONPATH. Running inside the project venv ensures `ml_playground` is importable.
- Quality tooling is mandatory before commit (ruff, mypy, pyright), and tests must pass.
- Linear history for own work: rebase your branches and avoid merge commits; fast-forward only. See DEVELOPMENT.md → “Git Workflow: Linear history”.
- Test-Driven Development (TDD) is required for functional changes: write a failing test, implement minimal code to pass, then refactor.
- Granular commits are required. Each functional/behavioral change MUST pair its production code with the corresponding tests in the same commit (unit/integration). Exceptions: documentation-only, test-only refactors, and mechanical formatting.

Setup and Developer Workflow

- See `.dev-guidelines/SETUP.md` for environment setup, quality gates, and the TDD-first developer workflow.
- See `.dev-guidelines/DEVELOPMENT.md` for full development and testing policies.

Datasets

- Shakespeare (GPT-2 BPE; prepared via internal ml_playground.experiments.shakespeare)
- Bundestag (char-level; prepared via internal ml_playground.experiments.bundestag_char; requires a user-provided text at ml_playground/experiments/bundestag_char/datasets/input.txt)
- Bundestag (tiktoken BPE; prepared via internal ml_playground.experiments.bundestag_tiktoken)

Workflows (high-level)

- Prepare/train/sample workflows are driven by Make targets. For exact commands, refer to each experiment's `Readme.md` and `.dev-guidelines/SETUP.md`.

Notes

- Configuration is defined via TOML dataclasses (see `ml_playground/config.py`).
- CPU/MPS are first-class. CUDA may be selected in TOML if available.
- Checkpoint behavior and policies are described in `.dev-guidelines/DEVELOPMENT.md` and `.dev-guidelines/REQUIREMENTS.md`.
- For framework utilities, see [Framework Utilities Documentation](docs/framework_utilities.md).

Mutation testing

- See `.dev-guidelines/SETUP.md` for how to run optional mutation testing (Cosmic Ray).

Loop

- See `.dev-guidelines/SETUP.md` for end-to-end loop examples.

TensorBoard (auto-enabled)

- Training logs to TensorBoard. See `.dev-guidelines/SETUP.md` for commands.

GGUF export (vendor approach)

- See `tools/llama_cpp/README.md` for the exact steps.

Testing

- See `.dev-guidelines/DEVELOPMENT.md` for testing standards and gates.
- See `tests/*/README.md` for folder-specific scope and patterns.

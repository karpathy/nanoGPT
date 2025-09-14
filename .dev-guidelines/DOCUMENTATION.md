---
trigger: always_on
description: documentation standards for nanoGPT/ml_playground
---

# Documentation Guidelines

Unified standards for all documentation in this repository. Applies to top-level docs, module docs, experiment READMEs, tests READMEs, and tools.

## Abstraction Policy

- Top level (repo `README.md`): high-level orientation and entrypoints only.
- Mid level (e.g., `ml_playground/experiments/Readme.md`): overview of experiments and shared conventions. Avoid per-file details.
- Low level (e.g., `ml_playground/experiments/<name>/Readme.md`): operational detail to run the experiment, but remain concise. Defer general concepts to shared docs.
- The deeper you go in the tree, the more concrete the instructions — but keep them brief and avoid duplication.

## Required Sections per Experiment Readme

- Overview (bulleted): dataset/model/method/pipeline at a glance.
- Data (concise): point to relevant config keys (e.g., `[prepare]`, `[train.data]`).
- Method/Model (concise): point to `[train.*]` keys; avoid restating defaults.
- Environment Setup: the minimal commands (`make setup`, etc.).
- How to Run: short, copy-pastable commands for prepare/train/sample/loop.
- Configuration Highlights: only essential keys grouped by section; omit exhaustive TOML.
- Outputs: brief list of important artifacts/paths.
- Folder structure: include a tree with a short inline description after each entry.
- Troubleshooting and Notes: concise, only experiment-specific items.

## Folder Tree Standard

- Use a fenced code block with `bash` language for syntax highlighting.
- Each entry must include a short inline description using `#` comments after two or more spaces.
- Keep the tree small; avoid listing generated files except when critical to orientation.

Example:

```bash
ml_playground/experiments/shakespeare/
├── Readme.md        # experiment documentation (this file)
├── config.toml      # sample/preset config for real runs overwriting ml_playground/experiments/default_config.toml
├── preparer.py      # dataset preparation
├── trainer.py       # training orchestration
├── sampler.py       # generation/sampling entrypoints
└── datasets/        # prepared dataset artifacts
```

## Linking to Framework Docs

- When referring to shared helpers or patterns, link to `docs/framework_utilities.md` rather than duplicating explanations.
- Example line: “For framework utilities, see `docs/framework_utilities.md`.”

## Markdown Style (markdownlint)

- Headings (MD022): surround with one blank line above and below.
- Lists (MD032): one blank line before and after lists.
- Fenced code blocks (MD031): one blank line before and after; specify a language when possible (MD040).
- Avoid inline HTML where a Markdown alternative exists (MD033).
- Ensure a single trailing newline (MD047).

## Cross-Referencing

- Use relative links within the repository: `../../docs/framework_utilities.md` from experiment folders.
- Prefer short, stable link text.

## DRY Documentation

- Avoid repeating extensive “what each file does” prose—lean on the annotated folder tree.
- Avoid restating default configuration; point to config sections instead.
- Prefer one canonical place for shared narratives, and link to it.

## Commit and Review Expectations

- Docs must pass `make quality` (markdownlint part of our checks) before commit.
- Prefer granular commits with clear `docs(<area>): <subject>` messages.
- When documenting functional changes to code, ensure the associated commit pairs production code and tests in the same change per TDD policy (see `DEVELOPMENT.md`).
- Reviewers check: abstraction level appropriate, tree annotated, headings/lists/code blocks spacing correct, and links valid.

## Tests and Tools READMEs

- Tests (`tests/*/README.md`): keep to purpose, how to run, principles, folder structure. No deep internal details.
- Tools (e.g., `tools/llama_cpp/README.md`): short purpose, exact usage, and a small annotated tree.

## Maintenance

- When adding new experiments, ensure their README follows this guideline from the start.
- When refactoring layout, update folder trees and links in the same PR.

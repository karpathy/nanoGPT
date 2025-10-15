---
trigger: always_on
description: Standards for documentation structure, abstraction levels, and formatting
---

# Documentation Guidelines

Unified standards for all documentation in this repository. Applies to top-level docs, module docs, experiment
READMEs, tests READMEs, and tools.

<details>
<summary>Related documentation</summary>

- [Developer Guidelines Index](./Readme.md) – Entry point for the ml_playground guideline system with quick-start commands and core principles.
- [Development Practices](./DEVELOPMENT.md) – Core development practices, quality standards, and workflow for ml_playground contributors.

</details>

## Table of Contents

- [Abstraction Policy](#abstraction-policy)
- [Required Sections per Experiment Readme](#required-sections-per-experiment-readme)
- [Folder Tree Standard](#folder-tree-standard)
- [Linking to Framework Docs](#linking-to-framework-docs)
- [Markdown Style (mdformat)](#markdown-style-mdformat)
- [Guideline Divergence Documentation](#guideline-divergence-documentation)
- [Cross-Referencing](#cross-referencing)
- [Cross-Document Details Blocks](#cross-document-details-blocks)
- [DRY Documentation](#dry-documentation)
- [Brevity](#brevity)
- [Commit and Review Expectations](#commit-and-review-expectations)
- [Tests and Tools READMEs](#tests-and-tools-readmes)
- [Maintenance](#maintenance)

## Abstraction Policy

- Top level (repo `README.md`): high-level orientation and entrypoints only.
- Mid level (e.g., `ml_playground/experiments/Readme.md`): overview of experiments and shared conventions. Avoid per-file
  details.
- Low level (e.g., `ml_playground/experiments/<name>/Readme.md`): operational detail to run the experiment, but remain
  concise. Defer general concepts to shared docs.
- The deeper you go in the tree, the more concrete the instructions — but keep them brief and avoid duplication.

## Required Sections per Experiment Readme

- Follow the canonical blueprint documented in `ml_playground/experiments/Readme.md`.
- Keep content focused on experiment-specific context; link back to shared docs for
  general workflows, commands, or policies.
- Include only the sections that add new information beyond the shared overview, trimming
  any duplicated instructions.

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

- When referring to shared helpers or patterns, link to `docs/framework_utilities.md` rather than duplicating
  explanations.
- Example line: “For framework utilities, see `docs/framework_utilities.md`.”

## Markdown Style (mdformat)

- Run `uv run mdformat <file.md>` (or rely on `uv run ci-tasks quality`) to format Markdown consistently.
- Maintain one blank line above and below headings.
- Ensure a blank line before and after lists; mdformat enforces indentation automatically.
- Add a blank line before and after fenced code blocks and specify a language when possible.
- Prefer Markdown constructs over inline HTML.
- Files must end with a single trailing newline.

## Guideline Divergence Documentation

- Any code or documentation that must temporarily diverge from repository guidelines **must** include a nearby TODO
  comment in the format `# TODO Remove <context>: <reason>` (or language-appropriate equivalent).
- The TODO comment must summarize the violating rule and the concrete exit criteria so IDEs and tooling can surface it.
- Mention any linked task/PR when known, and remove the TODO as soon as the divergence is resolved.

## Cross-Referencing

- Use relative links within the repository: `../../docs/framework_utilities.md` from experiment folders.
- Prefer short, stable link text.
- When linking to another folder, link to that folder's `Readme.md` (single entry point) instead of deep files. Deep
  documents should be discovered from that folder's `Readme.md`.

## Cross-Document Details Blocks

- Place a `<details>` block titled `Related documentation` near the top of READMEs that reference other guidelines.
- Each list item inside the block must be a relative Markdown link whose summary sentence is derived from the first
  paragraph or block of text immediately following the referenced document's top-level heading.
- Keep the list minimal—link only to documents that provide indispensable context.
- Update the referenced document summary if its opening paragraph changes.

### Template

```markdown
<details>
<summary>Related documentation</summary>

- [Document Title](./DOCUMENTATION.md) – First-paragraph summary from the referenced document.

</details>
```

## DRY Documentation

- Avoid repeating extensive “what each file does” prose—lean on the annotated folder tree.
- Avoid restating default configuration; point to config sections instead.
- Prefer one canonical place for shared narratives, and link to it.

## Brevity

- Keep documents concise. Prefer bulleted lists over long paragraphs.
- Avoid duplicating content across files; link to the canonical source instead.
- Trim examples to the minimal set needed to illustrate usage.

## Commit and Review Expectations

- Docs must pass the markdownlint portion of our quality gates. The pre-commit hook runs `uv run ci-tasks quality`,
  so avoid bypassing it.
- Prefer granular commits with clear `docs(<area>): <subject>` messages.
- When documenting functional changes to code, ensure the associated commit pairs production code and tests in the same
  change per TDD policy (see `DEVELOPMENT.md`).
- Reviewers check: abstraction level appropriate, tree annotated, headings/lists/code blocks spacing correct, and links
  valid.

## Tests and Tools READMEs

- Tests (`tests/*/README.md`): keep to purpose, how to run, principles, folder structure. No deep internal details.
- Tools (e.g., `tools/llama_cpp/README.md`): short purpose, exact usage, and a small annotated tree.

## Maintenance

- When adding new experiments, ensure their README follows this guideline from the start.
- When refactoring layout, update folder trees and links in the same PR.

# Tiny Shakespeare (GPT-2 BPE)

Minimal experiment to prepare, train, and sample on the Tiny Shakespeare corpus using GPT-2 BPE tokenization.

## Overview

- Dataset: Tiny Shakespeare (auto-downloaded)
- Encoding: GPT-2 BPE via tiktoken
- Method: Classic NanoGPT-style training (strictly typed, TOML-configured)
- Pipeline: prepare → train → sample via ml_playground CLI

## Data

- Preparer auto-downloads `input.txt` if missing.
- Prepared artifacts under `[train.data].dataset_dir` (default: `ml_playground/experiments/shakespeare/datasets/`).

## Method/Model

- GPT-2 BPE via tiktoken; small GPT configured via TOML (see `[train.*]`).
- Rotated checkpoints and TensorBoard logs under `[train.runtime].out_dir`.
For framework utilities, see [../../docs/framework_utilities.md](../../docs/framework_utilities.md).

## How to Run

- Config: `ml_playground/experiments/shakespeare/config.toml`

```bash
# Prepare → Train → Sample (separate)
uv run prepare-shakespeare
uv run train-shakespeare-cpu
uv run sample-shakespeare-cpu

# Or end-to-end
uv run loop-shakespeare-cpu
```

## Configuration Highlights

- `[train.data].dataset_dir` default: `ml_playground/experiments/shakespeare/datasets`
- `[train.runtime].out_dir` default: `ml_playground/experiments/shakespeare/out/shakespeare_next`
- `[train.runtime].device`: `cpu` or `mps` (or `cuda` if available)

## Outputs

- Training artifacts under `[train.runtime].out_dir` (rotated checkpoints, `logs/tb`).
- Prepared data under `[train.data].dataset_dir` (`train.bin`, `val.bin`).

## Folder structure

```text
ml_playground/experiments/shakespeare/
├── Readme.md        - experiment documentation (this file)
├── config.toml      - sample/preset config for real runs
├── test_config.toml - tiny defaults for tests
├── preparer.py      - dataset preparation (download/tokenize, write bins/meta)
├── trainer.py       - NanoGPT-style training orchestration
├── sampler.py       - generation/sampling entrypoints
└── datasets/        - prepared dataset artifacts written here
```

## Troubleshooting

- If tokenization fails, ensure `tiktoken` is installed and accessible.

## Notes

- Prepared data is written only to this experiment's `datasets/` directory.

## Checklist

- Adheres to `.dev-guidelines/DOCUMENTATION.md` (abstraction, required sections).
- Folder tree includes inline descriptions for each entry.
- Links to shared docs where applicable (e.g., `../../docs/framework_utilities.md`).
- Commands are copy-pasteable and minimal (setup, prepare/train/sample/loop).
- Configuration Highlights only list essential keys; defaults are not restated.
- Outputs paths and filenames reflect current behavior (check `[train.runtime].out_dir`).

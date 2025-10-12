# Bundestag (Char-Level)

Character-level language modeling on Bundestag speeches with a simple vocabulary built from the dataset characters.

## Overview

- Dataset: Custom text (provide input.txt manually)
- Encoding: Per-character IDs (uint16)
- Method: Classic NanoGPT-style training (strict TOML config)
- Pipeline: prepare → train → sample via ml_playground CLI

## Data

- Input: src/ml_playground/experiments/bundestag_char/datasets/input.txt
  - Preparers fail if this file is missing; create it with your own text.
- Outputs (prepared):
  - train.bin, val.bin (uint16 arrays)
  - meta.pkl (vocab metadata with stoi/itos, vocab_size)

## Method/Model

- Build vocabulary from unique characters in the corpus
- Encode train/val splits 90/10 into uint16 arrays
- Model architecture and training hyperparameters are specified in TOML
- TensorBoard logging at out_dir/logs/tb
  This experiment uses the centralized framework utilities for error handling, progress reporting, and file operations. For more information, see [Framework Utilities Documentation](../../docs/framework_utilities.md).

## Environment Setup (UV-only)

```bash
uv run setup
```

## Strict configuration injection

- This experiment does not read TOML directly. The CLI loads and validates the TOML and injects config objects into the experiment code.

## How to Run

- Config example: src/ml_playground/experiments/bundestag_char/config.toml

Prepare:

```bash
uvx --from . dev-tasks prepare bundestag_char
```

Train:

```bash
uvx --from . dev-tasks train bundestag_char --config ml_playground/experiments/bundestag_char/config.toml
```

Sample:

```bash
uvx --from . dev-tasks sample bundestag_char --config ml_playground/experiments/bundestag_char/config.toml
```

End-to-end loop:

```bash
uvx --from . dev-tasks loop bundestag_char --config ml_playground/experiments/bundestag_char/config.toml
```

## Configuration Highlights

- \[train.data\]
  - dataset_dir = "src/ml_playground/experiments/bundestag_char/datasets"
  - train_bin = "train.bin", val_bin = "val.bin", meta_pkl = "meta.pkl"
  - batch_size, block_size, grad_accum_steps
- \[train.runtime\]
  - out_dir = "src/ml_playground/experiments/bundestag_char/out/bundestag_char_next"
  - device = "cpu" (or "mps"/"cuda" if available), dtype = "float32"
- \[sample.runtime\]
  - out_dir should match train.runtime.out_dir
- \[sample.sample\]
  - start prompt text, num_samples, max_new_tokens

## Outputs

- Data artifacts: src/ml_playground/experiments/bundestag_char/datasets/{train.bin,val.bin,meta.pkl}
- Training: out_dir contains rotated checkpoints only, e.g.:
  - ckpt_last_XXXXXXXX.pt
  - `ckpt_best_XXXXXXXX_<metric>.pt`
  - logs/tb

## Folder structure

```bash
src/ml_playground/experiments/bundestag_char/
├── Readme.md        # experiment documentation (this file)
├── config.toml      # sample/preset config for real runs
├── test_config.toml # tiny defaults for tests
├── preparer.py      # dataset preparation (char vocab, encode, write bins/meta)
├── trainer.py       # NanoGPT-style training orchestration
├── sampler.py       # generation/sampling entrypoints
├── ollama_export.py # GGUF/Ollama export helper for this experiment
├── datasets/        # prepared dataset artifacts written here
└── export/          # export artifacts directory (e.g., GGUF)
```

## Troubleshooting

- If sampling fails with a missing `meta.pkl`, ensure it exists at `[train.data].dataset_dir` alongside `train.bin` and `val.bin`, or under `[sample.runtime].out_dir/<experiment>/meta.pkl` as per the CLI discovery rules.
- Ensure your input text is UTF-8 encoded.

## Word Tokenizer Option

- This experiment now supports a word-level tokenizer in addition to char/n-gram.
- To enable, set in config under \[train.data\]:

```toml
# Tokenizer selection: "char" (default) or "word"
tokenizer = "word"
# ngram_size is ignored when tokenizer="word"
```

- Sampling and training automatically use the dataset's meta.pkl; decoding joins tokens with a single space.

## Checklist

- Adheres to [.dev-guidelines/Readme.md](../../.dev-guidelines/Readme.md) (abstraction, required sections).
- Folder tree includes inline descriptions for each entry.
- Links to shared docs where applicable (e.g., `../../docs/framework_utilities.md`).
- Commands are copy-pasteable and minimal (setup, prepare/train/sample/loop).
- Configuration Highlights only list essential keys; defaults are not restated.
- Outputs paths and filenames reflect current behavior (check `[train.runtime].out_dir`).

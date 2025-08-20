# Bundestag (Char-Level)

Character-level language modeling on Bundestag speeches with a simple vocabulary built from the dataset characters.

## Overview
- Dataset: Custom text (seeded with page1.txt by default)
- Encoding: Per-character IDs (uint16)
- Method: Classic NanoGPT-style training (strict TOML config)
- Pipeline: prepare → train → sample via ml_playground CLI

## Data
- Input: ml_playground/experiments/bundestag_char/datasets/page1.txt
  - If missing, the preparer seeds it from a bundled sample file; replace with your own text for real runs.
- Outputs (prepared):
  - train.bin, val.bin (uint16 arrays)
  - meta.pkl (vocab metadata with stoi/itos, vocab_size)

## Method/Model
- Build vocabulary from unique characters in the corpus
- Encode train/val splits 90/10 into uint16 arrays
- Model architecture and training hyperparameters are specified in TOML
- TensorBoard logging at out_dir/logs/tb

## Environment Setup (UV-only)
```bash
uv venv --clear
uv sync --all-groups
```

## How to Run
- Config example: ml_playground/experiments/bundestag_char/config.toml

Prepare:
```bash
uv run python -m ml_playground.cli prepare bundestag_char
```

Train:
```bash
uv run python -m ml_playground.cli train bundestag_char
```

Sample:
```bash
uv run python -m ml_playground.cli sample bundestag_char
```

End-to-end loop:
```bash
uv run python -m ml_playground.cli loop bundestag_char
```

## Configuration Highlights
- [train.data]
  - dataset_dir = "ml_playground/experiments/bundestag_char/datasets"
  - train_bin = "train.bin", val_bin = "val.bin", meta_pkl = "meta.pkl"
  - batch_size, block_size, grad_accum_steps
- [train.runtime]
  - out_dir = "ml_playground/experiments/bundestag_char/out/bundestag_char_next"
  - device = "cpu" (or "mps"/"cuda" if available), dtype = "float32"
- [sample.runtime]
  - out_dir should match train.runtime.out_dir
- [sample.sample]
  - start prompt text, num_samples, max_new_tokens

## Outputs
- Data artifacts: ml_playground/experiments/bundestag_char/datasets/{train.bin,val.bin,meta.pkl}
- Training: out_dir contains ckpt_best.pt, ckpt_last.pt, logs/tb

## Troubleshooting
- If meta.pkl is missing at sampling, the CLI loop copies it to out_dir automatically; otherwise, place meta.pkl next to checkpoints.
- Ensure your input text is UTF-8 encoded.

## Notes
- The preparer is registered in ml_playground.experiments and is invoked by CLI prepare bundestag_char.
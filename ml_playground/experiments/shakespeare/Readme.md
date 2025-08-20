# Tiny Shakespeare (GPT-2 BPE)

Minimal experiment to prepare, train, and sample on the Tiny Shakespeare corpus using GPT-2 BPE tokenization.

## Overview
- Dataset: Tiny Shakespeare (auto-downloaded)
- Encoding: GPT-2 BPE via tiktoken
- Method: Classic NanoGPT-style training (strictly typed, TOML-configured)
- Pipeline: prepare → train → sample via ml_playground CLI

## Data
- The preparer downloads input.txt if missing.
- Prepared files are written to:
  - ml_playground/experiments/shakespeare/datasets/{input.txt, train.bin, val.bin}

## Method/Model
- Tokenization: GPT-2 BPE (tiktoken)
- Model: Small GPT configured in TOML (n_layer, n_head, n_embd, block_size, etc.)
- Checkpoints: ckpt_best.pt and ckpt_last.pt managed by trainer
- Logging: TensorBoard enabled at out_dir/logs/tb

## Environment Setup (UV-only)
```bash
uv venv --clear
uv sync --all-groups
```

## How to Run
- Config example: ml_playground/experiments/shakespeare/config.toml

Prepare dataset:
```bash
uv run python -m ml_playground.cli prepare shakespeare
```

Train:
```bash
uv run python -m ml_playground.cli train ml_playground/experiments/shakespeare/config.toml
```

Sample:
```bash
uv run python -m ml_playground.cli sample ml_playground/experiments/shakespeare/config.toml
```

End-to-end loop:
```bash
uv run python -m ml_playground.cli loop shakespeare ml_playground/experiments/shakespeare/config.toml
```

## Configuration Highlights
- [train.data]
  - dataset_dir = "ml_playground/experiments/shakespeare/datasets"
  - batch_size, block_size, grad_accum_steps
- [train.runtime]
  - out_dir = "ml_playground/experiments/shakespeare/out/shakespeare_next"
  - device = "cpu" or "mps" (or "cuda" if available)
- [sample.runtime]
  - out_dir should match train.runtime.out_dir

## Outputs
- Training: out_dir contains ckpt_best.pt, ckpt_last.pt, logs/tb
- Data: ml_playground/experiments/shakespeare/datasets/{train.bin, val.bin}

## Troubleshooting
- If download fails, check internet connection or provide input.txt manually under ml_playground/experiments/shakespeare/datasets/
- If sampling shows tokenization issues, ensure tiktoken is installed and accessible in the environment

## Notes
- Dataset preparer for this experiment is registered in ml_playground.experiments.
- Prepared data is written only to this experiment's datasets/ directory.
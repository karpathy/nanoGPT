# Bundestag (tiktoken BPE)

Byte-Pair Encoding (BPE) experiment using tiktoken to tokenize Bundestag speeches into uint32 token IDs.

## Overview
- Dataset: Custom text provided as input.txt
- Encoding: tiktoken (default: cl100k_base)
- Method: NanoGPT-style training with strict TOML configuration
- Pipeline: prepare → train → sample via ml_playground CLI

## Data
- Input: ml_playground/experiments/bundestag_tiktoken/datasets/input.txt
  - If missing, you may place a sample input.txt there manually (the preparer can also seed from a bundled resource if present).
- Outputs (prepared):
  - train.bin, val.bin (uint32 arrays)
  - meta.pkl (tokenizer metadata: encoding name and dtype)

## Method/Model
- Split corpus 90/10 into train/val
- Tokenize with tiktoken (cl100k_base by default) into uint32 arrays
- Model hyperparameters and runtime behavior controlled by TOML
- TensorBoard logs written to out_dir/logs/tb

## Environment Setup (UV-only)
```bash
uv venv --clear
uv sync --all-groups
```

## How to Run
- Config example: ml_playground/experiments/bundestag_tiktoken/bundestag_tiktoken_cpu.toml

Prepare dataset:
```bash
uv run python -m ml_playground.cli prepare bundestag_tiktoken
```

Train:
```bash
uv run python -m ml_playground.cli train ml_playground/experiments/bundestag_tiktoken/bundestag_tiktoken_cpu.toml
```

Sample:
```bash
uv run python -m ml_playground.cli sample ml_playground/experiments/bundestag_tiktoken/bundestag_tiktoken_cpu.toml
```

End-to-end loop:
```bash
uv run python -m ml_playground.cli loop bundestag_tiktoken ml_playground/experiments/bundestag_tiktoken/bundestag_tiktoken_cpu.toml
```

## Configuration Highlights
- [train.data]
  - dataset_dir = "ml_playground/experiments/bundestag_tiktoken/datasets"
  - train_bin, val_bin, meta_pkl
  - batch_size, block_size, grad_accum_steps
- [train.runtime]
  - out_dir = "ml_playground/experiments/bundestag_tiktoken/out/bundestag_tiktoken"
  - device = "mps" (or "cpu"/"cuda")
- [sample.runtime]
  - out_dir should match train.runtime.out_dir

## Outputs
- Data artifacts: ml_playground/experiments/bundestag_tiktoken/datasets/{train.bin,val.bin,meta.pkl}
- Training: out_dir contains checkpoints and logs/tb

## Troubleshooting
- Ensure tiktoken is installed; otherwise, the sampler will fall back to a byte codec.
- If you see ID dtype mismatches, verify that meta.pkl indicates "uint32" and that your training config expects corresponding sizes.

## Notes
- The preparer is registered in ml_playground.experiments and invoked via CLI prepare bundestag_tiktoken.
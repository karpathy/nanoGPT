# Bundestag Finetuning (Qwen2.5‑1.5B + LoRA on MPS)

Preset of the generic HF+PEFT integration to fine‑tune Qwen2.5‑1.5B on Bundestag speeches. Uses LoRA adapters and is optimized for Apple Silicon (MPS). CUDA also supported.

## Overview
- Model: Qwen/Qwen2.5‑1.5B (swapable via config)
- Dataset: Folder of .txt files or a single CSV; metadata can be preserved
- Method: Parameter‑efficient LoRA fine‑tuning (PEFT)
- Pipeline: prepare → train → sample via TOML‑driven integration

## Data
- `[prepare].raw_dir`: directory of `.txt` files (recursive) or a single CSV
- `[prepare].dataset_dir`: output location for tokenizer, train/val JSONL, meta
- Optional: `add_structure_tokens = true` to wrap speeches with SPEAKER/PARTY/YEAR tokens
- `doc_separator` controls a boundary token when packing documents

## Method/Model
- Loads tokenizer and base model from Hugging Face
- Applies LoRA to attention modules (and optionally MLP)
- Evaluation runs periodically; TensorBoard logs at `out_dir/logs/tb`
- Adapters saved to `out_dir/adapters/{best,last,final}`

## Environment Setup (UV-only)
```bash
uv venv --clear
uv sync --all-groups
# If needed, add runtime packages
uv add peft transformers torch tensorboard
```

## How to Run
This preset uses the `bundestag_finetuning_mps` integration under the hood.

End‑to‑end loop:
```bash
uv run python -m ml_playground.cli loop bundestag_qwen15b_lora_mps
```

Individual steps:
```bash
# Prepare
uv run python -m ml_playground.cli prepare bundestag_qwen15b_lora_mps

# Train
uv run python -m ml_playground.cli train bundestag_qwen15b_lora_mps

# Sample
uv run python -m ml_playground.cli sample bundestag_qwen15b_lora_mps
```

## Configuration Highlights
- `[prepare]`: `dataset = "bundestag_finetuning_mps"`, set `raw_dir`, `dataset_dir`, `add_structure_tokens`, `doc_separator`
- `[train.hf_model]`: `model_name = "Qwen/Qwen2.5-1.5B"`, `gradient_checkpointing = true`, `block_size = 256`
- `[train.peft]`: `enabled`, `r`, `lora_alpha`, `lora_dropout`, `bias`, `target_modules`, `extend_mlp_targets`
- `[train.data]`: `dataset_dir`, `batch_size`, `grad_accum_steps`, `block_size`, `shuffle`
- `[train.runtime]`: `out_dir`, `max_iters`, `eval_interval`, `eval_iters`, `device`, `dtype`, `compile`, checkpoint policy
- `[sample.runtime]`, `[sample.sample]` for generation

## Outputs
```
out/bundestag_qwen15b_lora_mps/
├── adapters/
│   ├── best/
│   ├── last/
│   └── final/
├── tokenizer/
├── samples/
│   └── sample-<timestamp>.txt
└── logs/
    └── tb/
```

## Troubleshooting
- Gated models/tokenizer downloads: set `HUGGINGFACE_HUB_TOKEN=hf_***` in `.env` or run `uv run huggingface-cli login`
- Cache downloads via `TRANSFORMERS_CACHE` (preferred) or `HUGGINGFACE_HUB_CACHE`/`HF_HOME`
- Memory constraints: lower `batch_size`, raise `grad_accum_steps`, reduce `block_size`, keep `gradient_checkpointing=true`

## Notes
- This preset rides on the generic integration at `ml_playground/experiments/bundestag_finetuning_mps`.
- Swap the base model by changing `[train.hf_model].model_name`.

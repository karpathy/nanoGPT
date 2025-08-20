# Bundestag Finetuning (HF + PEFT, MPS)

Generic Hugging Face + PEFT integration to fine-tune LLMs on Bundestag speeches with LoRA adapters. Optimized for Apple Silicon (MPS), works with CUDA as well.

## Overview
- Models: any causal LM from Hugging Face (e.g., Qwen, Gemma, Llama variants)
- Dataset: Folder of .txt files or a single CSV; metadata can be preserved
- Method: Parameter-efficient LoRA fine-tuning (PEFT)
- Pipeline: prepare → train → sample driven entirely by a TOML config

## Data
- `[prepare].raw_dir`: either a directory of `.txt` files (recursively scanned) or a single CSV file
- `[prepare].dataset_dir`: output location for tokenizer, JSONL splits, and metadata
- Optional structure tokens (`add_structure_tokens`) insert speaker/party/year tokens around speeches
- `doc_separator` controls a document boundary token used during packing

## Method/Model
- Loads tokenizer and base model from Hugging Face
- Applies LoRA adapters to attention (and optionally MLP) modules
- Packs/streams dataset efficiently; runs eval every N steps
- TensorBoard logging is enabled at out_dir/logs/tb
- Adapters/checkpoints are saved under out_dir/adapters/{best,last,final}

## Environment Setup (UV-only)
```bash
uv venv --clear
uv sync --all-groups
# If needed, add runtime packages for HF + PEFT
uv add peft transformers torch tensorboard
```

## How to Run
This integration is selected when `dataset == "bundestag_finetuning_mps"` or the config contains the integration blocks.

End-to-end loop (example using the Qwen preset):
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
- `[prepare]`
  - `dataset = "bundestag_finetuning_mps"`
  - `raw_dir`, `dataset_dir`, `add_structure_tokens`, `doc_separator`
- `[train.hf_model]`
  - `model_name` (e.g., `Qwen/Qwen2.5-1.5B`), `gradient_checkpointing`, `block_size`
- `[train.peft]`
  - `enabled`, `r`, `lora_alpha`, `lora_dropout`, `bias`, `target_modules`, `extend_mlp_targets`
- `[train.data]`
  - `dataset_dir`, `batch_size`, `grad_accum_steps`, `block_size`, `shuffle`
- `[train.runtime]`
  - `out_dir`, `max_iters`, `eval_interval`, `eval_iters`, `device`, `dtype`, `compile`
  - checkpoint policy fields like `ckpt_time_interval_minutes`, `keep_last_n`, `save_merged_on_best`
- `[sample.runtime]` and `[sample.sample]` for generation settings

## Outputs
```
out/<your_run_name>/
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
- Hugging Face gated models: place `HUGGINGFACE_HUB_TOKEN=hf_***` in a `.env` (auto-loaded by CLI) or login with `huggingface-cli login`
- Cache downloads by setting `TRANSFORMERS_CACHE` (preferred) or `HUGGINGFACE_HUB_CACHE`/`HF_HOME`
- MPS memory: reduce `batch_size`, increase `grad_accum_steps`, lower `block_size`, keep `gradient_checkpointing=true`

## Notes
- This directory contains the integration code. Example configs are provided in the experiment roots: `bundestag_qwen15b_lora_mps/config.toml` and `speakger/config.toml` for different base models.

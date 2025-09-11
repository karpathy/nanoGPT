# SpeakGer: Gemma Fine-Tuning Workflow (MPS, LoRA)

This document describes the Gemma 3 fine-tuning workflow for generating debates on current topics in the style of historic Bundestag speakers using the SpeakGer dataset. It follows a consistent blueprint used across experiments: Overview → Data → Method/Model → Environment Setup → How to Run → Configuration Highlights → Outputs → Troubleshooting → Notes.

## Overview

- Model Support: Google Gemma 3 models (2B, 9B variants)
- Dataset: SpeakGer with metadata preservation (speaker, party, year/era)
- Platform: Optimized for Apple Silicon (MPS)
- Method: LoRA (Low-Rank Adaptation) via PEFT
- Pipeline: prepare → train → sample driven by a TOML config

## Data

Minimal I/O with strict config injection (see `[prepare]` in TOML):

- `raw_dir`: either a folder of `.txt` files (SpeakGer-style) or a single CSV
- `dataset_dir`: where tokenizer/, JSONL, metadata, and splits are written
- CSVs are streamed (no full in-memory load). Basic columns like content/speaker/party/year/topic are auto-detected when present.

## Method/Model

- HF model + tokenizer (Gemma 2B/9B suggested)
- LoRA (PEFT) adapters for efficient finetuning
- Optional structure tokens for metadata preservation

For framework utilities, see [../../docs/framework_utilities.md](../../docs/framework_utilities.md).

## Environment Setup (UV-only)

```bash
uv run setup
# Install PEFT/Transformers if not already present in your env
uv add peft transformers torch tensorboard
```

## Strict configuration injection

- This experiment does not read TOML directly. The CLI loads and validates the TOML and injects config objects into the experiment code.

## How to Run

- Config: `ml_playground/experiments/speakger/config.toml`

```bash
# Prepare → Train → Sample (separate)
uv run prepare-speakger
uv run train-speakger
uv run sample-speakger

# Or end-to-end
uv run loop-speakger
```

## Configuration Highlights

Memory-friendly defaults for 32GB Apple Silicon are recommended.

- `[prepare]`
  - `raw_dir`: path to .txt folder or CSV file
  - `dataset_dir`: prepared dataset output directory
  - `add_structure_tokens`: wrap content with speaker/party/year tokens
  - `doc_separator`: separator token between documents

- `[train.hf_model]`
  - `model_name`: e.g., `google/gemma-2-2b` or `google/gemma-2-9b-it`
  - `gradient_checkpointing = true`
  - `block_size`: 256–512 typical on MPS

- `[train.peft]`
  - `enabled = true`, `r = 8..16`, `lora_alpha = 16`
  - `target_modules = ["q_proj","k_proj","v_proj","o_proj"]`
  - `extend_mlp_targets = false` (set true to include MLP projections)

- `[train.data]`
  - `dataset_dir`: same as in `[prepare]`
  - `batch_size`, `grad_accum_steps` control effective batch size
  - `block_size`: token sequence length per batch element

- `[train.runtime]`
  - `out_dir`: where adapters/checkpoints/logs go
  - `device`: `mps` (Apple), `cuda` (NVIDIA), or `cpu`
  - `dtype`: `float16` suggested on mps/cuda, `float32` for stability

- `[sample.runtime]`
  - `out_dir`, `device`, `dtype`

- `[sample.sample]`
  - `start`: prompt string or `FILE:path/to/prompt.txt`
  - `num_samples`, `max_new_tokens`, `temperature`, `top_k`, `top_p`

## Outputs

```text
out/speakger_gemma3_270m_lora_mps/
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

## Folder structure

```text
ml_playground/experiments/speakger/
├── Readme.md           - experiment documentation (this file)
├── config.toml         - sample/preset config for real runs
├── test_config.toml    - tiny defaults for tests
├── preparer.py         - dataset preparation (tokenizer, JSONL, meta)
├── trainer.py          - HF+PEFT LoRA training orchestration
├── sampler.py          - generation/sampling entrypoints
├── datasets/           - prepared dataset artifacts written here
└── raw/                - place raw .txt files or CSV here
```

## Troubleshooting

- Gated models: set `HUGGINGFACE_HUB_TOKEN=hf_***` or run `uv run huggingface-cli login`.
- Cache downloads: use `TRANSFORMERS_CACHE` (preferred).
- Memory/MPS: lower `batch_size`, raise `grad_accum_steps`, reduce `block_size`; use `device="mps"`.

## Notes

- The CLI recognizes this integration when `dataset == "gemma_finetuning_mps"` or the TOML contains the integration block.
- See `ml_playground/experiments/bundestag_finetuning_mps` for a generic HF+PEFT integration usable with other base models.

## Checklist

- Adheres to `.dev-guidelines/DOCUMENTATION.md` (abstraction, required sections).
- Folder tree includes inline descriptions for each entry.
- Links to shared docs where applicable (e.g., `../../docs/framework_utilities.md`).
- Commands are copy-pasteable and minimal (setup, prepare/train/sample/loop).
- Configuration Highlights only list essential keys; defaults are not restated.
- Outputs paths and filenames reflect current behavior (check `[train.runtime].out_dir`).

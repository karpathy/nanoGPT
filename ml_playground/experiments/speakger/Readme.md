# SpeakGer: Gemma Fine-Tuning Workflow (MPS, LoRA)

This document describes the Gemma 3 fine-tuning workflow for generating debates on current topics in the style of historic Bundestag speakers using the SpeakGer dataset. It follows a consistent blueprint used across experiments: Overview → Data → Method/Model → Environment Setup → How to Run → Configuration Highlights → Outputs → Troubleshooting → Notes.

## Overview
- Model Support: Google Gemma 3 models (2B, 9B variants)
- Dataset: SpeakGer with metadata preservation (speaker, party, year/era)
- Platform: Optimized for Apple Silicon (MPS)
- Method: LoRA (Low-Rank Adaptation) via PEFT
- Pipeline: prepare → train → sample driven by a TOML config

## Data
You can provide data in two ways:

1) Folder of .txt files (SpeakGer-style)
   - Place raw `.txt` files in a directory structure like:
   ```
   your_dataset_path/
   ├── speaker1_speech1.txt
   ├── speaker1_speech2.txt
   ├── speaker2_speech1.txt
   └── ...
   ```

2) Single CSV file (Bundestag.csv supported)
   - Point `raw_dir` directly to your CSV, e.g. `ml_playground/experiments/speakger/raw/Bundestag.csv`
   - The preparer streams the CSV (no full in‑memory load) and preserves metadata when available.
   - Recognized columns (best-effort):
     - content: text, speech, content, body, rede, speech_text, transcript, full_text, speechContent
     - speaker: speaker, redner, name, speaker_name, PersonName, person, speaker_fullname, firstname/lastname
     - party: party, partei, fraction, faction, parliamentary_group, Fraktion, party_long, party_short
     - date/year: date, datum, year, jahr, time, timestamp, session_date, speech_date (year auto-extracted)
     - topic: topic, title, subject, agenda_item, thema, heading

Configure these paths in the TOML under `[prepare]`:
- `raw_dir`: folder of .txt files or a single CSV file
- `dataset_dir`: output directory for tokenizer, JSONL, metadata, splits

## Method/Model
- Tokenizer and base model loaded from Hugging Face (Gemma 2B/9B suggested)
- Parameter-efficient fine-tuning with LoRA adapters (PEFT)
- Structured wrapping of speeches (optional) to preserve metadata tokens
- TensorBoard logging enabled automatically (out_dir/logs/tb)
- Checkpointing of adapters under out_dir/adapters/{best,last,final}

## Environment Setup (UV-only)
```bash
uv venv --clear
uv sync --all-groups
# Install PEFT/Transformers if not already present in your env
uv add peft transformers torch tensorboard
```

## How to Run
Example config (edit to your paths/model):
- `ml_playground/experiments/speakger/config.toml`

End-to-end pipeline (prepare → train → sample):
```bash
uv run python -m ml_playground.cli loop gemma_finetuning_mps \
  ml_playground/experiments/speakger/config.toml
```

Run individual steps:
```bash
# Prepare dataset
uv run python -m ml_playground.cli prepare gemma_finetuning_mps \
  ml_playground/experiments/speakger/config.toml

# Train
uv run python -m ml_playground.cli train \
  ml_playground/experiments/speakger/config.toml

# Sample
uv run python -m ml_playground.cli sample \
  ml_playground/experiments/speakger/config.toml
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
```
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

## Troubleshooting
- Gated models (Hugging Face):
  - Put your token in a `.env` in repo root or CWD:
    - `HUGGINGFACE_HUB_TOKEN=hf_********************************`
  - Or login persistently:
    - `uv run huggingface-cli login --token hf_********************************`
- Download caching:
  - Set `TRANSFORMERS_CACHE` (preferred) or `HUGGINGFACE_HUB_CACHE`/`HF_HOME`
- Memory issues:
  - Reduce `train.data.batch_size`, increase `grad_accum_steps`
  - Reduce `block_size` to 256/128
  - Keep `gradient_checkpointing=true`
- MPS issues:
  - Use `device="mps"`; keep `compile=false` if compilation causes issues

## Notes
- The CLI recognizes this integration when `dataset == "gemma_finetuning_mps"` or the TOML contains the integration block.
- See `ml_playground/experiments/bundestag_finetuning_mps` for a generic HF+PEFT integration usable with other base models.

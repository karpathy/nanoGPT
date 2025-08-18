# Gemma Fine-Tuning Workflow (SpeakGer, Local MPS)

This document describes the Gemma 3 fine-tuning workflow for generating debates on current topics in the style of historic Bundestag speakers using the SpeakGer dataset.

## Overview

The Gemma workflow mirrors the existing Qwen fine-tuning structure and provides:

- **Model Support**: Google Gemma 3 models (270M, 2B, 9B variants)
- **Dataset**: SpeakGer with metadata preservation (speaker, party, year/era)
- **Platform**: Optimized for Apple Silicon (M3, 32GB) with MPS acceleration
- **Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Pipeline**: Complete prepare → train → sample workflow

## Prerequisites

### Apple MPS Requirements

1. **Hardware**: Apple Silicon Mac (M1, M2, M3) with at least 16GB RAM (32GB recommended)
2. **Software**: 
   - macOS 12.3+ (for MPS support)
   - Python 3.9+
   - PyTorch with MPS support

### Environment Setup

```bash
# Create and activate environment
uv venv --clear
uv sync --all-groups

# Install PEFT for LoRA fine-tuning (if not already installed)
uv add peft transformers torch tensorboard
```

### Dataset Preparation

You can provide data in two ways:

1. Folder of .txt files (SpeakGer-style):
   - Place raw `.txt` files in a directory structure like:
   ```
   your_dataset_path/
   ├── speaker1_speech1.txt
   ├── speaker1_speech2.txt
   ├── speaker2_speech1.txt
   └── ...
   ```
2. Single CSV file (Bundestag.csv supported):
   - Point raw_dir directly to your CSV, e.g. `ml_playground/datasets/Bundestag.csv`
   - The preparer streams the CSV (no full in‑memory load) and preserves metadata when available.
   - Recognized columns (best-effort):
     - content: text, speech, content, body, rede, speech_text, transcript, full_text, speechContent
     - speaker: speaker, redner, name, speaker_name, PersonName, person, speaker_fullname, firstname/lastname
     - party: party, partei, fraction, faction, parliamentary_group, Fraktion, party_long, party_short
     - date/year: date, datum, year, jahr, time, timestamp, session_date, speech_date (year auto-extracted)
     - topic: topic, title, subject, agenda_item, thema, heading

## Quick Start

### 1. Configure Dataset Path

Edit the configuration file to point to your SpeakGer dataset:

```toml
# In ml_playground/configs/speakger_gemma3_270m_lora_mps.toml
[prepare]
# EITHER a folder of .txt files OR a single CSV file
# Example (CSV): raw_dir = "ml_playground/datasets/Bundestag.csv"
raw_dir = "path/to/your/speakger/dataset"  # UPDATE THIS PATH
```

### 2. Choose Model Size

The configuration defaults to Gemma 2B. To switch models, edit:

```toml
[train.hf_model]
# Options:
model_name = "google/gemma-2-2b"        # 2B params (recommended, default)
# model_name = "google/gemma-2-9b-it"   # 9B params (larger, may need smaller batch)
# model_name = "google/gemma-2-2b-it"   # 2B instruction-tuned variant
```

### 3. Run Complete Pipeline

**Single command for full workflow:**
```bash
uv run python -m ml_playground.cli loop gemma_finetuning_mps ml_playground/configs/speakger_gemma3_270m_lora_mps.toml
```

**Or run individual steps:**
```bash
# Prepare dataset
uv run python -m ml_playground.cli train ml_playground/configs/speakger_gemma3_270m_lora_mps.toml

# Train model  
uv run python -m ml_playground.cli train ml_playground/configs/speakger_gemma3_270m_lora_mps.toml

# Generate samples
uv run python -m ml_playground.cli sample ml_playground/configs/speakger_gemma3_270m_lora_mps.toml
```

## Configuration Options

### Memory Management (32GB RAM Optimization)

The default configuration is conservative for 32GB systems:

```toml
[train.data]
batch_size = 6           # Per-step batch size
grad_accum_steps = 10    # Effective batch size = 60
block_size = 512         # Sequence length

[train.hf_model]
gradient_checkpointing = true  # Reduces memory usage
```

### Model Size Adjustments

**For 270M model** (ultra-conservative):
```toml
[train.hf_model]
model_name = "google/gemma-2-2b"  # Actually use 2B as 270M not available yet

[train.data]
batch_size = 8
grad_accum_steps = 8
```

**For 9B model** (requires more memory):
```toml
[train.hf_model]
model_name = "google/gemma-2-9b-it"

[train.data]
batch_size = 2           # Reduce batch size
grad_accum_steps = 20    # Maintain effective batch size
block_size = 256         # Reduce sequence length
```

### LoRA Configuration

```toml
[train.peft]
r = 8                    # LoRA rank (8-16 recommended)
lora_alpha = 16         # Scaling factor
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention layers
extend_mlp_targets = false  # Set true to include MLP layers
```

## Expected Outputs

### Training Progress
```
[gemma_finetuning_mps] Starting training with config: ml_playground/configs/speakger_gemma3_270m_lora_mps.toml
[gemma_finetuning_mps] Model: google/gemma-2-2b
[gemma_finetuning_mps] Device: mps
[gemma_finetuning_mps] Loading tokenizer and model...
[gemma_finetuning_mps] Setting up LoRA...
trainable params: 4,194,304 || all params: 2,508,582,912 || trainable%: 0.1672
[gemma_finetuning_mps] Train samples: 1234, Val samples: 137
iter      0: loss 3.2456, lr 1.50e-05, tokens/sec 245.3
iter     25: loss 2.8901, lr 6.00e-04, tokens/sec 251.7
iter     50: loss 2.4567, lr 1.20e-03, tokens/sec 248.9
...
```

### Generated Samples
```
Sprecher: Dr. Alice Weidel (AfD)
Thema: Aktuelle politische Entwicklungen
Jahr: 2024

Meine Damen und Herren, ich möchte heute über die aktuellen 
politischen Entwicklungen sprechen, die unser Land bewegen...
[Generated text continues in the style of historic Bundestag debates]
```

### File Structure
```
out/speakger_gemma3_270m_lora_mps/
├── adapters/
│   ├── best/           # Best performing LoRA adapters
│   ├── last/           # Most recent checkpoint
│   └── final/          # Final training checkpoint
├── tokenizer/          # Saved tokenizer
├── samples/            # Generated samples
│   └── sample-1234567890.txt
└── logs/
    └── tb/             # TensorBoard logs
```

## Advanced Usage

### Custom Prompts

Create a custom prompt file:
```bash
echo "Sprecher: Angela Merkel (CDU)
Thema: Klimawandel
Jahr: 2019

Sehr geehrte Damen und Herren," > custom_prompt.txt
```

Update config:
```toml
[sample.sample]
start = "FILE:custom_prompt.txt"
```

### TensorBoard Monitoring

```bash
uv run tensorboard --logdir out/speakger_gemma3_270m_lora_mps/logs/tb --port 6006
# Open http://localhost:6006
```

### Batch Processing

Process multiple configurations:
```bash
for config in configs/gemma_*.toml; do
    uv run python -m ml_playground.cli loop gemma_finetuning_mps "$config"
done
```

## Platform Notes

### Apple MPS (Primary Target)
- **Pros**: Native acceleration, good memory efficiency, no CUDA setup required
- **Cons**: Limited to Apple hardware, some operations fall back to CPU
- **Recommended**: Default choice for Apple Silicon Macs

### NVIDIA CUDA (Alternative)
```toml
[train.runtime]
device = "cuda"

[sample.runtime]
device = "cuda"
```
- **Pros**: Full GPU acceleration, wider hardware support
- **Cons**: Requires CUDA setup, may need different memory settings

### Google TPU (Cloud Alternative)
```toml
[train.runtime]
device = "tpu"  # Requires TPU-specific PyTorch setup
```
- **Note**: Requires significant configuration changes and cloud setup

## Troubleshooting

### Memory Issues
1. **Reduce batch_size**: Start with `batch_size = 2`
2. **Increase grad_accum_steps**: Maintain effective batch size
3. **Reduce block_size**: Use `block_size = 256` or `128`
4. **Enable gradient_checkpointing**: Should be `true` by default

### MPS Issues
1. **Fallback to CPU**: Set `device = "cpu"` if MPS causes problems
2. **Compilation errors**: Keep `compile = false` for MPS
3. **Memory spikes**: Monitor Activity Monitor during training

### Dataset Issues
1. **No text files found**: Check `raw_dir` path in config
2. **Encoding errors**: Ensure text files are UTF-8 encoded
3. **Empty samples**: Verify text files contain actual content

### Model Loading Issues
1. **Model not found**: Check internet connection for HuggingFace download
2. **Authentication**: Login with `huggingface-hub login` if needed for gated models
3. **Disk space**: Ensure sufficient space for model downloads (2-20GB per model)

## Performance Expectations

### Apple M3 32GB Estimates
- **Gemma 2B**: ~150-250 tokens/sec, ~2-4 hours for 10k iterations
- **Gemma 9B**: ~50-100 tokens/sec, ~8-12 hours for 10k iterations
- **Memory usage**: 60-80% of 32GB during training

### Quality Expectations
- **Initial training**: Coherent German text after ~1000 iterations
- **Style adaptation**: Bundestag-style rhetoric after ~3000 iterations  
- **Speaker conditioning**: Metadata-aware generation after ~5000 iterations

## Examples

See `ml_playground/configs/speakger_gemma3_270m_lora_mps.toml` for a complete, annotated configuration example.

---

*This workflow provides a streamlined path from raw SpeakGer data to fine-tuned Gemma models optimized for local Apple Silicon development.*

## Gated models (Hugging Face authentication)

You can place your token in a local .env file and it will be auto‑loaded by the CLI:

- Create a file named .env either in the project root (next to pyproject.toml) or in your current working directory.
- Add a line: HUGGINGFACE_HUB_TOKEN=hf_********************************

Some Gemma weights are gated on Hugging Face (for example: google/gemma-2-2b). If you encounter an error like:

OSError: You are trying to access a gated repo (401 Unauthorized)

authenticate with your Hugging Face token. No changes to the TOML are needed; the integration automatically picks up authentication.

Recommended UV-only options:

- Temporary for current shell/session (no prompt):
  export HUGGINGFACE_HUB_TOKEN=hf_********************************
  uv run python -m ml_playground.cli loop gemma_finetuning_mps ml_playground/configs/speakger_gemma3_270m_lora_mps.toml

- Persist login in the venv (recommended):
  uv run huggingface-cli login --token hf_********************************
  # or programmatically:
  uv run python -c "from huggingface_hub import login; login(token='hf_********************************')"

Verify login (optional):
  uv run python -c "from huggingface_hub import whoami; print(whoami())"

Notes:
- You must have been granted access to the gated repo (visit the model page and request access if needed).
- If you still see a 401, re-check your token and that it has the correct scope. The CLI will print a helpful hint.
- To remove a persistent login: uv run huggingface-cli logout

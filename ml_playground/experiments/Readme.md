# Experiments (Mid‑Level Overview)

This directory hosts self‑contained experiments. Each experiment bundles:
- its data preparation logic (prepare.py),
- a TOML config (at the experiment root),
- a local datasets/ area for seeds and prepared artifacts,
- a focused Readme.md with step‑by‑step instructions,
- and, where applicable, trainer/integration code.

Why self‑contained?
- Portability: copy a single folder to reuse an experiment.
- Reproducibility: config, code, and sample data paths live together.
- Discoverability: each experiment explains itself in its own Readme.
- Decoupling: no dependency on the legacy ml_playground/datasets package (the CLI falls back to experiment registries).

Conventions
- Registration: experiment preparers register via ml_playground.experiments.register. The CLI discovers them automatically.
- Names: the experiment argument to `prepare`/`loop` equals the experiment’s registered name.
- Config location: TOML lives at the experiment root (no configs/ subfolder).
- Data location: experiment‑local prepared data lives under `<experiment>/datasets/`.
- Outputs: example configs write to `<experiment>/out/<run_name>`.
- Typing/UV: everything follows the project’s strict typing and UV‑only workflow (see repo README for commands).

Common CLI patterns
- Prepare: `uv run python -m ml_playground.cli prepare <experiment_name>`
- Train: `uv run python -m ml_playground.cli train <experiment_name>`
- Sample: `uv run python -m ml_playground.cli sample <experiment_name>`
- End‑to‑end: `uv run python -m ml_playground.cli loop <experiment_name>`

Implemented experiments (current)
- shakespeare — Tiny Shakespeare with GPT‑2 BPE (tiktoken)
  - Readme: ml_playground/experiments/shakespeare/Readme.md
  - Config:  ml_playground/experiments/shakespeare/config.toml
  - Prepare name: `shakespeare`
- bundestag_char — Character‑level modeling on Bundestag text
  - Readme: ml_playground/experiments/bundestag_char/Readme.md
  - Config:  ml_playground/experiments/bundestag_char/config.toml
  - Prepare name: `bundestag_char`
- bundestag_tiktoken — BPE tokenization (tiktoken) for Bundestag text
  - Readme: ml_playground/experiments/bundestag_tiktoken/Readme.md
  - Config:  ml_playground/experiments/bundestag_tiktoken/config.toml
  - Prepare name: `bundestag_tiktoken`
- bundestag_finetuning_mps — Generic HF + PEFT LoRA finetuning integration (Apple MPS‑friendly)
  - Readme: ml_playground/experiments/bundestag_finetuning_mps/Readme.md
  - Example preset config: ml_playground/experiments/bundestag_qwen15b_lora_mps/config.toml
  - Dataset value in TOML/CLI: `bundestag_finetuning_mps`
- bundestag_qwen15b_lora_mps — Qwen2.5‑1.5B preset for the generic finetuning integration
  - Readme: ml_playground/experiments/bundestag_qwen15b_lora_mps/Readme.md
  - Config:  ml_playground/experiments/bundestag_qwen15b_lora_mps/config.toml
  - Uses dataset/integration: `bundestag_finetuning_mps`
- speakger — Gemma‑based finetuning workflow targeting SpeakGer‑style data
  - Readme: ml_playground/experiments/speakger/Readme.md
  - Config:  ml_playground/experiments/speakger/config.toml
  - Uses dataset/integration: `gemma_finetuning_mps` (see notes in the experiment Readme)

Add a new experiment (checklist)
1) Create a folder: `ml_playground/experiments/<name>/`
2) Implement data prep in `prepare.py` and register it:
   ```python
   from ml_playground.experiments import register

   @register("<name>")
   def main() -> None:
       # your preparation logic
       ...
   ```
3) Place seeds and prepared artifacts in `<name>/datasets/` (created at runtime).
4) Put a TOML config at `<name>/config.toml`, referenced by your README and examples.
5) Write `<name>/Readme.md` following the common blueprint: Overview → Data → Method/Model → Environment → How to Run → Config Highlights → Outputs → Troubleshooting → Notes.

Notes
- The CLI first tries to import legacy `ml_playground.datasets.PREPARERS`; if absent, it uses `ml_playground.experiments.PREPARERS` (this directory). This lets you delete the legacy datasets package without breaking the CLI.
- Keep paths inside configs relative to the repo for portability.


## New Experiment Template
Use this copy-ready template to create a new experiment at `ml_playground/experiments/<name>/`.

- Files to create:
  - `ml_playground/experiments/<name>/prepare.py`
  - `ml_playground/experiments/<name>/config.toml`
  - `ml_playground/experiments/<name>/Readme.md`
  - `ml_playground/experiments/<name>/datasets/` (created at runtime)

Paste the following into `ml_playground/experiments/<name>/Readme.md` and replace placeholders in angle brackets <> with your experiment specifics.

```markdown
# &lt;Title of Your Experiment&gt;

&lt;One-sentence summary of the goal. Example: "Minimal experiment to prepare, train, and sample on &lt;dataset&gt; using &lt;tokenization/method&gt;."&gt;

## Overview
- Dataset: &lt;dataset name/description and how it's obtained&gt;
- Encoding/Tokenizer: &lt;character-level | GPT-2 BPE via tiktoken | HF tokenizer name&gt;
- Method: &lt;NanoGPT-style training | HF + PEFT LoRA | custom&gt;
- Pipeline: prepare → train → sample via ml_playground CLI

## Data
- Inputs (raw): &lt;path(s) under this experiment, e.g., `ml_playground/experiments/&lt;name&gt;/datasets/input.txt` or a `raw/` folder&gt;
  - &lt;If missing, describe seeding behavior or how to place inputs&gt;
- Outputs (prepared):
  - &lt;list prepared artifacts, e.g., train.bin, val.bin, meta.pkl or tokenizer/, train.jsonl, val.jsonl&gt;

## Method/Model
- &lt;Briefly describe tokenization/model. Example: "Tokenization: GPT-2 BPE (tiktoken). Model: small GPT configured via TOML (n_layer, n_head, n_embd, block_size)."&gt;
- Checkpoints: &lt;e.g., ckpt_best.pt, ckpt_last.pt or adapters under out_dir/adapters/{best,last,final}&gt;
- Logging: TensorBoard at `out_dir/logs/tb`

## Environment Setup (UV-only)
```bash
uv venv --clear
uv sync --all-groups
# Optional: add runtime deps if needed by your integration
# uv add peft transformers torch tensorboard
```

## How to Run
- Config path: `ml_playground/experiments/&lt;name&gt;/config.toml`

Prepare dataset:
```bash
uv run python -m ml_playground.cli prepare &lt;dataset_name&gt;
# &lt;dataset_name&gt; is the registered name in prepare.py (often "&lt;name&gt;")
```

Train:
```bash
uv run python -m ml_playground.cli train <dataset_name>
```

Sample:
```bash
uv run python -m ml_playground.cli sample <dataset_name>
```

End-to-end loop:
```bash
uv run python -m ml_playground.cli loop <dataset_name>
```

## Configuration Highlights
- [prepare]
  - `dataset = "&lt;dataset_name&gt;"`
  - `raw_dir`, `dataset_dir` &lt;describe what they point to&gt;
  - &lt;add integration-specific keys like `add_structure_tokens`, `doc_separator`&gt;
- [train.data]
  - `dataset_dir` and core knobs: `batch_size`, `block_size`, `grad_accum_steps`
- [train.runtime]
  - `out_dir = "ml_playground/experiments/&lt;name&gt;/out/&lt;run_name&gt;"`
  - `device = "cpu" | "mps" | "cuda"`, `dtype = "float32" | "float16"`
- [sample.runtime]
  - `out_dir` should match `train.runtime.out_dir`
- [sample.sample]
  - `start` prompt, `num_samples`, `max_new_tokens`, `temperature`, `top_k`, `top_p`

## Outputs
- Data artifacts: `ml_playground/experiments/&lt;name&gt;/datasets/...`
- Training artifacts: `out_dir` contains checkpoints (and adapters if applicable) and `logs/tb`

## Troubleshooting
- &lt;Common issues and fixes&gt;
- &lt;e.g., if sampling shows tokenization issues, ensure tiktoken is installed; for HF gated models, set HUGGINGFACE_HUB_TOKEN in .env&gt;

## Notes
- The dataset preparer for this experiment should be registered in `ml_playground.experiments` and invoked by the CLI.
- Keep all paths in TOML relative to the repo root for portability.
```

Example `prepare.py`:
```python
from __future__ import annotations
from pathlib import Path
from ml_playground.experiments import register

@register("&lt;dataset_name&gt;")
def main() -> None:
    """Prepare the &lt;name&gt; dataset."""
    config_path = Path(__file__).parent / "config.toml"
    # Call your integration or custom code here.
    # e.g., integ.prepare_from_toml(config_path)
    print(f"[&lt;name&gt;.prepare] Using config: {config_path}")
```

Example `config.toml` (adapt to your integration):
```toml
[prepare]
dataset = "&lt;dataset_name&gt;"
raw_dir = "ml_playground/experiments/&lt;name&gt;/raw"
dataset_dir = "ml_playground/experiments/&lt;name&gt;/datasets"

[train.data]
dataset_dir = "ml_playground/experiments/&lt;name&gt;/datasets"
batch_size = 4
block_size = 128
grad_accum_steps = 1

[train.runtime]
out_dir = "ml_playground/experiments/&lt;name&gt;/out/&lt;name&gt;_run"
device = "cpu"
dtype = "float32"
seed = 1

[sample.runtime]
out_dir = "ml_playground/experiments/&lt;name&gt;/out/&lt;name&gt;_run"
device = "cpu"
dtype = "float32"
seed = 1
compile = false

[sample.sample]
start = "Hello"
max_new_tokens = 64
temperature = 0.8
top_k = 200
top_p = 0.95
num_samples = 1
```

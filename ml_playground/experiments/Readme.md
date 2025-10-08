# Experiments (Mid‑Level Overview)

This directory hosts self‑contained experiments. Each experiment bundles:

- its data preparation logic (`preparer.py`),
- a TOML config (at the experiment root),
- a local datasets/ area for seeds and prepared artifacts,
- a focused Readme.md with step‑by‑step instructions,
- and, where applicable, trainer/integration code.

Why self‑contained?

- Portability: copy a single folder to reuse an experiment.
- Reproducibility: config, code, and sample data paths live together.
- Discoverability: each experiment explains itself in its own Readme.
- Decoupling: no dependency on any legacy `ml_playground/datasets` package (the CLI uses the experiment registry only).

Conventions

- Discovery: an experiment must expose a `preparer.py` with a class that implements a `.prepare(...)` method. The CLI auto‑discovers these preparers.
- Names: the experiment argument to `prepare`/`loop` equals the experiment’s registered name.
- Config location: TOML lives at the experiment root (no configs/ subfolder).
- Data location: experiment‑local prepared data lives under `<experiment>/datasets/`.
- Outputs: example configs write to `<experiment>/out/<run_name>`.
- Typing/UV: everything follows the project’s strict typing and UV‑only workflow (see repo README for commands).

Universal meta policy

- Preparers must write a standardized `meta.pkl` next to `train.bin` and `val.bin` in the dataset directory.
- Use `ml_playground.data_pipeline.write_bin_and_meta(...)` to atomically write `train.bin`, `val.bin`, and `meta.pkl`.
- The CLI will fail fast at train/sample time if `meta.pkl` is missing.

All experiments now use the centralized framework utilities for error handling, progress reporting, and file operations. For more information, see [Framework Utilities Documentation](../docs/framework_utilities.md).

## Folder structure (this directory)

```bash
ml_playground/experiments/
├── Readme.md                    # overview and conventions (this file)
├── __init__.py                  # package marker/registry setup
├── default_config.toml          # baseline config used by templates/examples
├── protocol.py                  # typed protocol/contracts for preparers
├── shakespeare/                 # Tiny Shakespeare experiment
├── bundestag_char/              # Bundestag character-level experiment
├── bundestag_tiktoken/          # Bundestag BPE (tiktoken) experiment
├── bundestag_qwen15b_lora_mps/  # Qwen2.5 LoRA preset over generic integration
└── speakger/                    # Gemma-based fine-tuning workflow (SpeakGer)
```

Important: Strict configuration injection

- Experiments must not read TOML directly. The CLI reads TOML and injects fully validated config objects into experiment code.
- Any legacy helpers like `prepare_from_toml`, `train_from_toml`, `sample_from_toml`, or `convert_from_toml` have been removed or fail fast.

Common CLI patterns

- Prepare: `uvx --from . dev-tasks prepare <name> [--config path]` (e.g., `uvx --from . dev-tasks prepare shakespeare`).
- Train: `uvx --from . dev-tasks train <name> --config path` (e.g., `uvx --from . dev-tasks train shakespeare --config ml_playground/configs/shakespeare_cpu.toml`).
- Sample: `uvx --from . dev-tasks sample <name> --config path` (e.g., `uvx --from . dev-tasks sample shakespeare --config ml_playground/configs/shakespeare_cpu.toml`).
- End‑to‑end: `uvx --from . dev-tasks loop <name> --config path` (e.g., `uvx --from . dev-tasks loop bundestag_char --config ml_playground/configs/bundestag_char_cpu.toml`).

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
2) Implement a strict preparer in `preparer.py` that exposes a class with `.prepare(...)` (see template below).

3) Place seeds and prepared artifacts in `<name>/datasets/` (created at runtime).
4) Put a TOML config at `<name>/config.toml`, referenced by your README and examples.
5) Write `<name>/Readme.md` following the common blueprint: Overview → Data → Method/Model → Environment → How to Run → Config Highlights → Outputs → Troubleshooting → Notes.

Notes

- The CLI uses `ml_playground.experiments.PREPARERS` (auto‑discovered). Legacy `ml_playground/datasets` registries are not used.
- Keep paths inside configs relative to the repo for portability.

## New Experiment Template

Use this copy-ready template to create a new experiment at `ml_playground/experiments/<name>/`.

- Files to create:

  - `ml_playground/experiments/<name>/preparer.py`
  - `ml_playground/experiments/<name>/config.toml`
  - `ml_playground/experiments/<name>/Readme.md`
  - `ml_playground/experiments/<name>/datasets/` (created at runtime)

Paste the following into `ml_playground/experiments/<name>/Readme.md` and replace placeholders in angle brackets <> with your experiment specifics.

```markdown
# <Title of Your Experiment>

<One-sentence summary of the goal. Example: "Minimal experiment to prepare, train, and sample on <dataset> using <tokenization/method>.">

## Overview
- Dataset: <dataset name/description and how it's obtained>
- Encoding/Tokenizer: <character-level | GPT-2 BPE via tiktoken | HF tokenizer name>
- Method: <NanoGPT-style training | HF + PEFT LoRA | custom>
- Pipeline: prepare → train → sample via ml_playground CLI

## Data
- Inputs (raw): <path(s) under this experiment, e.g., `ml_playground/experiments/<name>/datasets/input.txt` or a `raw/` folder>
  - <If missing, describe seeding behavior or how to place inputs>
- Outputs (prepared):
  - <list prepared artifacts, e.g., train.bin, val.bin, meta.pkl or tokenizer/, train.jsonl, val.jsonl>

## Method/Model
- <Briefly describe tokenization/model. Example: "Tokenization: GPT-2 BPE (tiktoken). Model: small GPT configured via TOML (n_layer, n_head, n_embd, block_size)." >
- Checkpoints (rotated-only):
  - `ckpt_last_XXXXXXXX.pt`
  - `ckpt_best_XXXXXXXX_<metric>.pt`
- Logging: TensorBoard at `out_dir/logs/tb`

## Environment Setup (preferred)

```bash
uvx --from . dev-tasks setup
uvx --from . dev-tasks verify
```

## How to Run

- Config path: `ml_playground/experiments/<name>/config.toml`

Prepare dataset:

```bash
uvx --from . dev-tasks prepare <name>
```

Train:

```bash
uvx --from . dev-tasks train <name> --config ml_playground/experiments/<name>/config.toml
```

Sample:

```bash
uvx --from . dev-tasks sample <name> --config ml_playground/experiments/<name>/config.toml
```

End-to-end loop:

```bash
uvx --from . dev-tasks loop <name> --config ml_playground/experiments/<name>/config.toml
```

## Configuration

- Use the default settings in `ml_playground/configs/` where available and adjust via experiment configs under `ml_playground/experiments/<name>/config.toml`.
- For tiny test defaults, see `tests/e2e/ml_playground/experiments/test_default_config.toml`.

## Outputs

- Data artifacts: `ml_playground/experiments/<name>/datasets/...`
- Training artifacts: `out_dir` contains checkpoints (and adapters if applicable) and `logs/tb`

## Troubleshooting

- Common issues and fixes
- e.g., if sampling shows tokenization issues, ensure tiktoken is installed; for HF gated models, set HUGGINGFACE_HUB_TOKEN in .env

## Notes

- The dataset preparer for this experiment is auto‑discovered via a class in `preparer.py` and invoked by the CLI.
- Keep all paths in TOML relative to the repo root for portability.
Example `preparer.py` (strict API):

```python
from __future__ import annotations
from pathlib import Path
from ml_playground.configuration import PreparerConfig
from ml_playground.data_pipeline import write_bin_and_meta
from ml_playground.experiments.protocol import Preparer as _PreparerProto, PrepareReport

class MyPreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        exp_dir = Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        # ... your preparation logic ...
        # write_bin_and_meta(ds_dir, train_ids, val_ids, meta)
        return PrepareReport(created_files=(), updated_files=(), skipped_files=(), messages=("ok",))

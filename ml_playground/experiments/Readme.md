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
- Names: the dataset argument to `prepare`/`loop` equals the experiment’s registered name.
- Config location: TOML lives at the experiment root (no configs/ subfolder).
- Data location: experiment‑local prepared data lives under `<experiment>/datasets/`.
- Outputs: example configs write to `<experiment>/out/<run_name>`.
- Typing/UV: everything follows the project’s strict typing and UV‑only workflow (see repo README for commands).

Common CLI patterns
- Prepare: `uv run python -m ml_playground.cli prepare <dataset_name>`
- Train: `uv run python -m ml_playground.cli train <path/to/config.toml>`
- Sample: `uv run python -m ml_playground.cli sample <path/to/config.toml>`
- End‑to‑end: `uv run python -m ml_playground.cli loop <dataset_name> <path/to/config.toml>`

Implemented experiments (current)
- shakespeare — Tiny Shakespeare with GPT‑2 BPE (tiktoken)
  - Readme: ml_playground/experiments/shakespeare/Readme.md
  - Config:  ml_playground/experiments/shakespeare/shakespeare_cpu.toml
  - Prepare name: `shakespeare`
- bundestag_char — Character‑level modeling on Bundestag text
  - Readme: ml_playground/experiments/bundestag_char/Readme.md
  - Config:  ml_playground/experiments/bundestag_char/bundestag_char_cpu.toml
  - Prepare name: `bundestag_char`
- bundestag_tiktoken — BPE tokenization (tiktoken) for Bundestag text
  - Readme: ml_playground/experiments/bundestag_tiktoken/Readme.md
  - Config:  ml_playground/experiments/bundestag_tiktoken/bundestag_tiktoken_cpu.toml
  - Prepare name: `bundestag_tiktoken`
- bundestag_finetuning_mps — Generic HF + PEFT LoRA finetuning integration (Apple MPS‑friendly)
  - Readme: ml_playground/experiments/bundestag_finetuning_mps/Readme.md
  - Example preset config: ml_playground/experiments/bundestag_qwen15b_lora_mps/bundestag_qwen15b_lora_mps.toml
  - Dataset value in TOML/CLI: `bundestag_finetuning_mps`
- bundestag_qwen15b_lora_mps — Qwen2.5‑1.5B preset for the generic finetuning integration
  - Readme: ml_playground/experiments/bundestag_qwen15b_lora_mps/Readme.md
  - Config:  ml_playground/experiments/bundestag_qwen15b_lora_mps/bundestag_qwen15b_lora_mps.toml
  - Uses dataset/integration: `bundestag_finetuning_mps`
- speakger — Gemma‑based finetuning workflow targeting SpeakGer‑style data
  - Readme: ml_playground/experiments/speakger/Readme.md
  - Config:  ml_playground/experiments/speakger/speakger_gemma3_270m_lora_mps.toml
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
4) Put a TOML config at `<name>/<name>_cpu.toml` (or similar), referenced by your README and examples.
5) Write `<name>/Readme.md` following the common blueprint: Overview → Data → Method/Model → Environment → How to Run → Config Highlights → Outputs → Troubleshooting → Notes.

Notes
- The CLI first tries to import legacy `ml_playground.datasets.PREPARERS`; if absent, it uses `ml_playground.experiments.PREPARERS` (this directory). This lets you delete the legacy datasets package without breaking the CLI.
- Keep paths inside configs relative to the repo for portability.

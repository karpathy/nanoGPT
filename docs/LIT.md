# LIT (Learning Interpretability Tool) Integration

This repository integrates Google's Learning Interpretability Tool (LIT) as an optional, self-contained analysis module under `ml_playground/analysis`. LIT provides an interactive UI for per-example and aggregate analysis of models.

References:

- Overview and docs: <https://pair-code.github.io/lit/>
- Getting Started (datasets, models, API, running server): <https://pair-code.github.io/lit/documentation/getting_started.html>
- GitHub: <https://github.com/PAIR-code/lit>
- PyPI: <https://pypi.org/project/lit-nlp/>

## What you can do

- Explore a tiny built-in text sample from the bundestag_char experiment (10–50 examples).
- Run a trivial echo model adapter to see LIT inspectors in action.
- Extend by plugging your own dataset or model adapter with minimal boilerplate.

## Install (UV-only)

This project uses UV exclusively. LIT is optional. Install it via the dedicated script:

```bash
make lit-setup
```

If you prefer manual installation, install `lit-nlp` in a compatible environment following the upstream docs.

Note: Some transitive dependencies (e.g., llvmlite via umap-learn/pynndescent) may not support every Python version. On Python 3.13, the optional `lit` extra is intentionally a no-op to avoid incompatible packages. If you need to use LIT, create a separate virtual environment with a compatible Python (<3.13) and install `lit-nlp` there, or skip this integration.

Python version: 3.13.x

## Run the analysis server (LIT UI)

Launch the local LIT UI for the bundestag_char PoC (default host 127.0.0.1, port 5432):

```bash
make lit PORT=5432
```

Flags:

- `--host` (default `127.0.0.1`)
- `--port` (default `5432`; use `0` for OS-assigned free port)
- `--open-browser` (open a browser tab automatically)

Expected output:

- Printed registered model(s) and dataset(s)
- URL to open in your browser (e.g., `http://127.0.0.1:5432`)

## Modules

- `ml_playground/analysis/lit_integration.py` — Minimal LIT integration for the bundestag_char PoC, with a tiny dataset and echo model.
- `ml_playground/cli.py` — Provides the `analyze` command to start the LIT server.

## Add your own

Follow the LIT Getting Started guide and mirror the patterns here:

- Implement a `Dataset` with `spec()` and an iterator over examples.
- Implement a `Model` adapter with `input_spec()`, `output_spec()`, and `predict()` mapping your model I/O to LIT fields.
- Register them in an analysis launcher and expose via the CLI if desired.

## Troubleshooting

- `ModuleNotFoundError: lit_nlp`: Install optional extra: `uv sync --extra lit` (or `uv add lit-nlp`).
- Port already in use: Pick a different port via `--port` or pass `--port 0` for an automatically chosen free port.
- No models/datasets in UI: Ensure `ml_playground/analysis/lit_integration.py` registers your model/dataset names and the CLI is launching `analyze`.
- GPU/CUDA: The demo adapter is CPU-only and trivial. Do not assume CUDA. If you add your own models, ensure CPU fallback is supported.

## Security & Privacy

- No telemetry or external calls beyond hosting the local UI.
- Sample dataset is embedded and safe.
- Do not bundle private checkpoints. If you add paths to models, make them opt-in flags and handle missing files gracefully.

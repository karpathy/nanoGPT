# LIT (Learning Interpretability Tool) — Minimal Integration

The optional LIT UI lives under `ml_playground/analysis/lit/` and provides a tiny echo-model demo for quick inspection. We keep this minimal and isolated from the main project environment.

## Quickstart (UV, Python 3.12)

```bash
# one-time setup
make venv312-lit-setup

# run the minimal LIT server
make venv312-lit-run PORT=5432

# stop (if needed)
make venv312-lit-stop PORT=5432
```

Open: [http://127.0.0.1:5432](http://127.0.0.1:5432)

Notes:

- Uses a dedicated `.venv312` with known-good pins (see `requirements/lit-demos.constraints.txt`).
- Caches datasets and downloads under `.cache/`.
- Intended as a thin scaffolding for your own adapters later.

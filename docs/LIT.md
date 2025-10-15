# LIT (Learning Interpretability Tool) — Minimal Integration

The optional LIT UI lives under `src/ml_playground/analysis/lit/` and provides a tiny echo-model demo
for quick inspection. We keep this minimal and isolated from the main project environment.

## Quickstart (UV, Python 3.12)

```bash
# one-time setup
uv run dev-tasks lit setup --python-version 3.12

# run the minimal LIT server
uv run dev-tasks lit run --port 5432

# stop (if needed)
uv run dev-tasks lit stop --port 5432
```

Open: [http://127.0.0.1:5432](http://127.0.0.1:5432)

Notes:

- Uses a dedicated `.venv312` with known-good pins (see `src/ml_playground/analysis/lit/requirements.txt`).
- Caches datasets and downloads under `.cache/`.
- Intended as a thin scaffolding for your own adapters later.

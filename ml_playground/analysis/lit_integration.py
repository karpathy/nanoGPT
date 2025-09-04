from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def run_server_bundestag_char(
    host: str = "127.0.0.1", port: int = 5432, open_browser: bool = False
) -> None:
    """Launch a minimal LIT server for the bundestag_char PoC.

    This uses a tiny embedded text dataset and a trivial echo model to
    demonstrate the LIT UI without requiring trained checkpoints.
    """
    try:
        # Lazy imports to avoid hard-dependency unless the command is used.
        from lit_nlp.api import dataset as lit_dataset  # type: ignore
        from lit_nlp.api import model as lit_model  # type: ignore
        from lit_nlp.api import types as lit_types  # type: ignore
        from lit_nlp import server as lit_server  # type: ignore
    except ImportError as e:  # pragma: no cover - import-guard path
        raise RuntimeError(
            "LIT is not available. Install the optional dependency first:\n"
            "  uv sync --extra lit\n"
            "or explicitly:\n"
            "  uv add lit-nlp\n"
            "See docs/LIT.md for details."
        ) from e

    # --- Tiny sample dataset ---
    # Prefer a few lines from the bundestag_char seed if present; otherwise use embedded samples.
    samples: List[str] = [
        "Nächste Rednerin ist die Vorsitzende der AfD-Fraktion, Dr. Alice Weidel.",
        "Herr Präsident, liebe Kolleginnen und Kollegen, wir beraten heute wichtige Vorlagen.",
        "(Beifall bei der SPD)",
        "Die Bundesregierung handelt entschlossen.",
        "Applaus bei der CDU/CSU.",
        "Vielen Dank. — Zur Geschäftsordnung hat der Abgeordnete das Wort.",
        "Wir müssen die Inflation bekämpfen und Familien entlasten.",
        "Das Wort hat nun die Bundeskanzlerin.",
        "Meine Damen und Herren, die Lage ist ernst, aber beherrschbar.",
        "(Heiterkeit) Der nächste Redner folgt.",
    ]

    # Try to read input.txt if it exists, but keep it optional and tiny.
    try:
        # Resolve to the ml_playground/experiments/bundestag_char directory
        base_dir = Path(__file__).resolve().parents[1]
        exp_dir = base_dir / "experiments" / "bundestag_char"
        input = exp_dir / "datasets" / "input.txt"
        if input.exists():
            text = input.read_text(encoding="utf-8", errors="ignore")
            # Take up to 10 reasonably short lines.
            file_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if file_lines:
                samples = file_lines[:10]
    except Exception:
        # Non-fatal; keep embedded samples
        pass

    class BundestagTextDataset(lit_dataset.Dataset):
        def __init__(self, sents: Iterable[str]):
            self._examples: List[Mapping[str, str]] = [{"text": s} for s in sents]

        def spec(self) -> Dict[str, object]:  # type: ignore[override]
            return {
                "text": lit_types.TextSegment(),
            }

        def __len__(self) -> int:
            return len(self._examples)

        def __iter__(self):
            return iter(self._examples)

    class EchoModel(lit_model.Model):
        """Trivial model that returns the input text as generated output.

        Serves as a PoC to exercise LIT views for text data without trained weights.
        """

        def input_spec(self) -> Dict[str, object]:  # type: ignore[override]
            return {"text": lit_types.TextSegment()}

        def output_spec(self) -> Dict[str, object]:  # type: ignore[override]
            # Use TextSegment for broad compatibility; some LIT versions also have GeneratedText.
            return {"generated": lit_types.TextSegment()}

        def predict(
            self, _inputs: Iterable[Mapping[str, object]], **kwargs: object
        ) -> List[Mapping[str, object]]:
            outs: List[Mapping[str, object]] = []
            for ex in _inputs:
                s = str(ex.get("text", ""))
                # Simple deterministic transform to show change
                gen = s + "\n\n[echo] " + s[::-1]
                outs.append({"generated": gen})
            return outs

    datasets = {"bundestag_char_sample": BundestagTextDataset(samples)}
    models = {"echo_model": EchoModel()}

    try:
        app = lit_server.Server(models, datasets)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to build LIT app: {e}") from e

    url = f"http://{host}:{port if port else '<auto>'}"
    print("[LIT] Registered models:", ", ".join(models.keys()))
    print("[LIT] Registered datasets:", ", ".join(datasets.keys()))
    print(f"[LIT] Starting server at {url}")
    sys.stdout.flush()

    try:
        lit_server.serve(app, port=port, host=host, open_browser=open_browser)
    except TypeError:
        lit_server.serve(app, port, host)

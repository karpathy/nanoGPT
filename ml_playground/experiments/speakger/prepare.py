from __future__ import annotations

from pathlib import Path
from typing import Optional

from ml_playground.experiments import register
from . import gemma_finetuning_mps as integ


@register("speakger")
def main() -> None:
    """Prepare the SpeakGer dataset via the Gemma finetuning integration.

    - Uses the experiment-local default TOML configuration.
    - If the configured raw_dir is a directory and has no .txt files, seed a tiny
      example text file so the integration can run out of the box.
    """
    config_path: Path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        raise SystemExit(
            f"Default config not found for speakger experiment: {config_path}"
        )

    # Best-effort: read raw_dir from TOML to decide whether seeding is needed
    raw_dir: Optional[Path] = None
    try:
        import tomllib  # Python 3.11+

        with open(config_path, "rb") as f:
            d = tomllib.load(f)
        p = d.get("prepare", {})
        raw_dir_str = p.get("raw_dir")
        if isinstance(raw_dir_str, str) and raw_dir_str.strip():
            raw_dir = Path(raw_dir_str)
    except Exception:
        # Non-fatal; fall back to default path used in the example config
        raw_dir = Path("ml_playground/experiments/speakger/raw")

    # If raw_dir points to a CSV file, the integration will handle it; skip seeding.
    # Otherwise, if raw_dir is a directory with no .txt files, write a tiny example.
    if raw_dir is not None:
        if raw_dir.suffix.lower() != ".csv":
            # Treat as a directory path
            raw_dir.mkdir(parents=True, exist_ok=True)
            has_txt = any(raw_dir.rglob("*.txt"))
            if not has_txt:
                example_path = raw_dir / "example_speech.txt"
                example_text = (
                    "Sprecher: Max Mustermann (BEISPIEL)\n"
                    "Thema: Beispielrede zur Pipelineprüfung\n"
                    "Jahr: 2024\n\n"
                    "Meine Damen und Herren, dies ist eine kurze Beispielrede, "
                    "um den Datensatzaufbau zu demonstrieren. Vielen Dank.\n"
                )
                try:
                    example_path.write_text(example_text, encoding="utf-8")
                    print(
                        f"[speakger.prepare] Seeded example text at {example_path} because no .txt files were found in {raw_dir}."
                    )
                except Exception:
                    # Seeding is best-effort; continue to integration regardless
                    pass

    # Delegate the actual preparation to the integration
    integ.prepare_from_toml(config_path)

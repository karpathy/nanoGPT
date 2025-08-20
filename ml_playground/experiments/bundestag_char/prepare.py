from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path
import pickle
import numpy as np
from ml_playground.experiments import register


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.uint16)


def prepare_from_text(text: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split into train/val and return arrays and meta dict.

    Meta contains vocab_size, stoi, itos.
    """
    n = len(text)
    train_data = text[: int(n * 0.9)]
    val_data = text[int(n * 0.9) :]
    stoi, itos = build_vocab(text)
    train_ids = encode_text(train_data, stoi)
    val_ids = encode_text(val_data, stoi)
    meta = {"vocab_size": len(stoi), "itos": itos, "stoi": stoi}
    return train_ids, val_ids, meta


@register("bundestag_char")
def main() -> None:
    """Prepare a character-level dataset for experiments/bundestag_char.

    - Reads page1.txt (prefers experiments/bundestag_char/datasets/page1.txt, with fallback to bundled resource)
    - Splits 90/10 train/val, builds vocab and encodes
    - Writes train.bin, val.bin, meta.pkl under experiments/bundestag_char/datasets
    """
    ds_dir = Path("ml_playground") / "experiments" / "bundestag_char" / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "page1.txt"

    if not input_file_path.exists():
        # Prefer experiment-local seed if available, fall back to bundled package resource
        exp_candidates = [
            Path("ml_playground")
            / "experiments"
            / "bundestag_char"
            / "datasets"
            / "page1.txt",
            Path("ml_playground") / "experiments" / "bundestag_char" / "page1.txt",
        ]
        src: Path | None = None
        for p in exp_candidates:
            if p.exists():
                src = p
                break
        if src is None:
            bundled = Path(__file__).parent / "page1.txt"
            if bundled.exists():
                src = bundled
        if src is not None:
            input_file_path.write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8"
            )
            print(f"Seeded {input_file_path} from resource {src}.")
        else:
            raise SystemExit(
                f"Expected input text at {input_file_path}, and no resource was found in experiments or bundled package."
            )

    data = input_file_path.read_text(encoding="utf-8")
    print(f"length of dataset in characters: {len(data):,}")

    train_ids, val_ids, meta = prepare_from_text(data)
    print("vocab size:", meta["vocab_size"])

    (ds_dir / "train.bin").write_bytes(train_ids.tobytes())
    (ds_dir / "val.bin").write_bytes(val_ids.tobytes())

    with (ds_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)  # type: ignore[arg-type]

    print("Wrote:", ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl")

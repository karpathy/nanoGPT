from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

from ml_playground.experiments import register
from ml_playground.prepare import (
    seed_text_file,
    split_train_val,
    encode_split_with_encoder,
    write_bin_and_meta,
    Encoder,
)


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


class CharEncoder:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi

    def encode_ordinary(self, text: str) -> list[int]:  # satisfies Encoder Protocol
        return [self.stoi[c] for c in text]


@register("bundestag_char")
def main() -> None:
    """Prepare a character-level dataset for experiments/bundestag_char.

    - Reads page1.txt (prefers experiment-local candidates, with fallback to bundled resource)
    - Splits 90/10 train/val, builds vocab and encodes using centralized helpers
    - Writes train.bin, val.bin, meta.pkl under experiments/bundestag_char/datasets (strict meta)
    """
    ds_dir = Path("ml_playground") / "experiments" / "bundestag_char" / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "page1.txt"

    # Seed input file from first available candidate; fail fast if none.
    bundled = Path(__file__).parent / "page1.txt"
    candidates = [
        Path("ml_playground") / "experiments" / "bundestag_char" / "datasets" / "page1.txt",
        Path("ml_playground") / "experiments" / "bundestag_char" / "page1.txt",
        bundled,
    ]
    seed_text_file(input_file_path, candidates)

    data = input_file_path.read_text(encoding="utf-8")
    print(f"length of dataset in characters: {len(data):,}")

    stoi, itos = build_vocab(data)
    enc: Encoder = CharEncoder(stoi)

    train_text, val_text = split_train_val(data, 0.9)
    train_arr, val_arr = encode_split_with_encoder(train_text, val_text, enc)

    meta = {
        "meta_version": 1,
        "kind": "char",
        "dtype": "uint16",
        "stoi": stoi,
        "itos": itos,
    }

    write_bin_and_meta(ds_dir, train_arr, val_arr, meta)

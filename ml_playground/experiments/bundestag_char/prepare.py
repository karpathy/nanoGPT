from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

from ml_playground.experiments import register
from ml_playground.prepare import (
    seed_text_file,
    split_train_val,
    write_bin_and_meta,
)
from ml_playground.config import load_toml
import numpy as np


def build_vocab(text: str, n: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    if n <= 1:
        tokens = sorted(set(text))
    else:
        tokens = sorted({text[i : i + n] for i in range(0, max(0, len(text) - n + 1))})
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for i, tok in enumerate(tokens)}
    return stoi, itos


def _encode_ngrams(text: str, stoi: Dict[str, int], n: int) -> list[int]:
    if n <= 1:
        return [stoi[c] for c in text]
    L = len(text)
    if L < n:
        return []
    ids: list[int] = []
    for i in range(0, L - n + 1):
        tok = text[i : i + n]
        idx = stoi.get(tok)
        if idx is not None:
            ids.append(idx)
    return ids


@register("bundestag_char")
def main() -> None:
    """Prepare a char/ngram-level dataset for experiments/bundestag_char.

    - Reads page1.txt (prefers experiment-local candidates, with fallback to bundled resource)
    - Splits 90/10 train/val, builds vocab from full text and encodes with stride-1 n-grams
    - Writes train.bin, val.bin, meta.pkl under experiments/bundestag_char/datasets (strict meta)
    """
    exp_dir = Path("ml_playground") / "experiments" / "bundestag_char"
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "input.txt"

    # Seed input file from first available candidate; fail fast if none.
    bundled = Path(__file__).parent / "input.txt"
    candidates = [
        ds_dir / "input.txt",
        exp_dir / "page1.txt",
        bundled,
    ]
    seed_text_file(input_file_path, candidates)

    data = input_file_path.read_text(encoding="utf-8")
    print(f"length of dataset in characters: {len(data):,}")

    # Read n-gram size from experiment config if available
    n: int = 1
    try:
        cfg_path = exp_dir / "config.toml"
        if cfg_path.exists():
            app = load_toml(cfg_path)
            if app.train is not None and app.train.data is not None:
                n = int(getattr(app.train.data, "ngram_size", 1))
    except Exception:
        n = 1
    if n < 1:
        n = 1

    stoi, itos = build_vocab(data, n)

    train_text, val_text = split_train_val(data, 0.9)
    train_ids = _encode_ngrams(train_text, stoi, n)
    val_ids = _encode_ngrams(val_text, stoi, n)

    vocab_size = int(len(stoi))
    use_uint16 = vocab_size <= 65535
    dtype_str = "uint16" if use_uint16 else "uint32"
    np_dtype = np.uint16 if use_uint16 else np.uint32

    train_arr = np.array(train_ids, dtype=np_dtype)
    val_arr = np.array(val_ids, dtype=np_dtype)

    meta = {
        "meta_version": 1,
        "kind": "char" if n == 1 else "char_ngram",
        "dtype": dtype_str,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }
    if n != 1:
        meta["ngram_size"] = n

    write_bin_and_meta(ds_dir, train_arr, val_arr, meta)

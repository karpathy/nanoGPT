from __future__ import annotations

from pathlib import Path
import numpy as np
import tiktoken

from ml_playground.experiments import register
from ml_playground.prepare import (
    seed_text_file,
    split_train_val,
    write_bin_and_meta,
)


@register("bundestag_tiktoken")
def main() -> None:
    """Prepare a BPE-tokenized dataset for experiments/bundestag_tiktoken using centralized helpers and strict meta.

    - Reads input.txt (prefers experiment-local candidates, with fallback to bundled resource)
    - Splits 90/10 into train/val
    - Encodes using tiktoken BPE and writes uint32 arrays
    - Writes train.bin, val.bin, and meta.pkl under experiments/bundestag_tiktoken/datasets
    """
    ds_dir = Path("ml_playground") / "experiments" / "bundestag_tiktoken" / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "input.txt"

    # Seed input file from first available candidate; fail fast if none.
    bundled = Path(__file__).parent / "input.txt"
    candidates = [
        Path("ml_playground") / "experiments" / "bundestag_tiktoken" / "datasets" / "input.txt",
        Path("ml_playground") / "experiments" / "bundestag_tiktoken" / "input.txt",
        bundled,
    ]
    seed_text_file(input_file_path, candidates)

    data = input_file_path.read_text(encoding="utf-8")
    print(f"length of dataset in characters: {len(data):,}")

    enc_name = "cl100k_base"
    enc = tiktoken.get_encoding(enc_name)

    train_text, val_text = split_train_val(data, 0.9)
    train_ids = enc.encode(train_text, allowed_special={"<|endoftext|>"})
    val_ids = enc.encode(val_text, allowed_special={"<|endoftext|>"})

    train_arr = np.array(train_ids, dtype=np.uint32)
    val_arr = np.array(val_ids, dtype=np.uint32)

    meta = {
        "meta_version": 1,
        "kind": "tiktoken",
        "dtype": "uint32",
        "encoding": enc_name,
    }

    write_bin_and_meta(ds_dir, train_arr, val_arr, meta)

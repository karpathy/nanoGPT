from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import tiktoken
from ml_playground.datasets import register


@register("bundestag_tiktoken")
def main() -> None:
    """Prepare a character-level dataset from ml_playground/datasets/bundestag_tiktoken/input.txt

    - Reads input.txt from ml_playground/datasets/bundestag_tiktoken
    - Splits 90/10 into train/val
    - Builds char-level vocab and encodes to uint16 ids
    - Writes train.bin, val.bin, and meta.pkl (stoi/itos/vocab_size)
    """
    ds_dir = Path("ml_playground") / "datasets" / "bundestag_tiktoken"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "input.txt"

    if not input_file_path.exists():
        # Auto-seed from bundled resource if the input file is missing
        bundled = Path(__file__).parent / "bundestag_tiktoken" / "input.txt"
        if bundled.exists():
            input_file_path.write_text(
                bundled.read_text(encoding="utf-8"), encoding="utf-8"
            )
            print(f"Seeded {input_file_path} from bundled resource {bundled}.")
        else:
            raise SystemExit(
                f"Expected input text at {input_file_path}, and no bundled resource was found at {bundled}."
            )

    data = input_file_path.read_text(encoding="utf-8")
    print(f"length of dataset in characters: {len(data):,}")

    data = input_file_path.read_text(encoding="utf-8")
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    enc_name = "cl100k_base"
    enc = tiktoken.get_encoding(enc_name)
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Use np.uint32, as token IDs may exceed 65535 for modern BPE vocabularies
    train_arr = np.array(train_ids, dtype=np.uint32)
    val_arr = np.array(val_ids, dtype=np.uint32)
    (ds_dir / "train.bin").write_bytes(train_arr.tobytes())
    (ds_dir / "val.bin").write_bytes(val_arr.tobytes())

    # Persist tokenizer metadata so sampling matches training.
    meta = {
        "encoding": enc_name,
        "dtype": "uint32",
    }
    with (ds_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    print("Wrote:", ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl")

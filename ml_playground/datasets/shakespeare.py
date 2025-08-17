from __future__ import annotations
from pathlib import Path
import requests
import numpy as np
import tiktoken
from ml_playground.datasets import register


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@register("shakespeare")
def main() -> None:
    """Prepare Tiny Shakespeare with GPT-2 BPE into ml_playground/datasets/shakespeare/*.bin

    - Downloads input.txt if missing
    - Splits 90/10 train/val
    - Encodes with tiktoken GPT-2 BPE
    - Writes uint16 train.bin and val.bin
    """
    ds_dir = Path("ml_playground") / "datasets" / "shakespeare"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "input.txt"

    if not input_file_path.exists():
        print(f"Downloading Tiny Shakespeare to {input_file_path}...")
        resp = requests.get(DATA_URL, timeout=60)
        resp.raise_for_status()
        input_file_path.write_text(resp.text, encoding="utf-8")

    data = input_file_path.read_text(encoding="utf-8")
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)
    (ds_dir / "train.bin").write_bytes(train_arr.tobytes())
    (ds_dir / "val.bin").write_bytes(val_arr.tobytes())

    # No meta.pkl needed for BPE case; sampler falls back to tiktoken if none provided
    print("Wrote:", ds_dir / "train.bin", ds_dir / "val.bin")

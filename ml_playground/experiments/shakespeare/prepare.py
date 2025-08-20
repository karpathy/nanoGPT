from __future__ import annotations

from pathlib import Path
import numpy as np
import requests
import tiktoken

from ml_playground.experiments import register
from ml_playground.prepare import split_train_val, write_bin_and_meta


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@register("shakespeare")
def main() -> None:
    """Prepare Tiny Shakespeare via GPT-2 BPE with strict meta and centralized helpers.

    Writes only to the experiment-local datasets directory:
    ml_playground/experiments/shakespeare/datasets/{input.txt, train.bin, val.bin, meta.pkl}

    Idempotent behavior: if train.bin and val.bin exist and are newer than input.txt,
    preparation is skipped.
    """
    # Resolve experiment-local datasets directory
    exp_ds_dir = Path(__file__).resolve().parent / "datasets"
    exp_ds_dir.mkdir(parents=True, exist_ok=True)

    f_input = exp_ds_dir / "input.txt"
    f_train = exp_ds_dir / "train.bin"
    f_val = exp_ds_dir / "val.bin"

    # Download input if missing
    if not f_input.exists():
        print(f"Downloading Tiny Shakespeare to {f_input}...")
        resp = requests.get(DATA_URL, timeout=60)
        resp.raise_for_status()
        f_input.write_text(resp.text, encoding="utf-8")

    # If outputs exist and are up-to-date relative to input, skip work
    try:
        if f_train.exists() and f_val.exists():
            t_in = f_input.stat().st_mtime
            if f_train.stat().st_mtime >= t_in and f_val.stat().st_mtime >= t_in:
                print("Already prepared:", f_train, f_val)
                return
    except OSError:
        # If any stat fails, proceed to regenerate
        pass

    # Read, split, and encode with tiktoken (GPT-2)
    data = f_input.read_text(encoding="utf-8")
    enc_name = "gpt2"
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

    # Write arrays and strict meta
    write_bin_and_meta(exp_ds_dir, train_arr, val_arr, meta)

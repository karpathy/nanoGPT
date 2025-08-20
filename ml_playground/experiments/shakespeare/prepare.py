from __future__ import annotations

from typing import Tuple, Protocol
from pathlib import Path
import numpy as np
import requests
import tiktoken
from ml_playground.experiments import register


class Encoder(Protocol):
    def encode_ordinary(self, text: str) -> list[int]: ...


def encode_split_with_encoder(
    train_data: str, val_data: str, enc: Encoder
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode provided train/val splits using a provided encoder.

    Returns two numpy arrays of dtype uint16.
    """
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)
    return train_arr, val_arr


def prepare_with_encoder(text: str, enc: Encoder) -> Tuple[np.ndarray, np.ndarray]:
    """Split text 90/10 and encode with a provided encoder."""
    n = len(text)
    train_data = text[: int(n * 0.9)]
    val_data = text[int(n * 0.9) :]
    return encode_split_with_encoder(train_data, val_data, enc)


def prepare_from_text(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split text 90/10 and encode with GPT-2 BPE (self-contained usage)."""
    enc = tiktoken.get_encoding("gpt2")
    return prepare_with_encoder(text, enc)


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@register("shakespeare")
def main() -> None:
    """Prepare Tiny Shakespeare via GPT-2 BPE.

    Writes only to the experiment-local datasets directory:
    ml_playground/experiments/shakespeare/datasets/{input.txt, train.bin, val.bin}
    (no writes to legacy ./datasets or repo root files)
    
    Idempotent behavior: if train.bin and val.bin exist and are newer than input.txt,
    preparation is skipped and a message is printed indicating data is up-to-date.
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

    # Read, encode, and split
    data = f_input.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("gpt2")
    train_arr, val_arr = prepare_with_encoder(data, enc)

    # Write train/val arrays to experiment-local datasets directory
    f_train.write_bytes(train_arr.tobytes())
    f_val.write_bytes(val_arr.tobytes())

    print("Wrote:", f_train, f_val)

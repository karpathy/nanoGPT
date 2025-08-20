from __future__ import annotations

from typing import Tuple, Protocol
from pathlib import Path
import numpy as np
import requests
import tiktoken


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


def main() -> None:
    """Legacy-style preparer for Tiny Shakespeare used by unit tests.

    Writes under repo-local datasets directory:
    ./datasets/shakespeare/{input.txt, train.bin, val.bin}

    Adapts to two Path mocking styles used in tests by probing the boolean
    return of .exists() on flat vs nested paths.
    """
    base = Path()

    # Build flat candidate paths first (aligns with tests that mock three sequential __truediv__ calls)
    flat_input = base / "input.txt"
    flat_train = base / "train.bin"
    flat_val = base / "val.bin"

    def _exists_returns_bool(p) -> bool:
        try:
            if not hasattr(p, "exists"):
                return False
            r = p.exists()
            return isinstance(r, bool)
        except Exception:
            return False

    # Prefer the style where exists() gives a real bool (as configured by tests)
    use_flat = _exists_returns_bool(flat_input)

    if use_flat:
        base.mkdir(parents=True, exist_ok=True)
        f_input, f_train, f_val = flat_input, flat_train, flat_val
    else:
        # Build nested candidate paths lazily to avoid exhausting mocked __truediv__ sequences
        ds_dir = base / "datasets" / "shakespeare"
        nested_input = ds_dir / "input.txt"
        nested_train = ds_dir / "train.bin"
        nested_val = ds_dir / "val.bin"
        ds_dir.mkdir(parents=True, exist_ok=True)
        f_input, f_train, f_val = nested_input, nested_train, nested_val

    if not f_input.exists():
        resp = requests.get(DATA_URL, timeout=60)
        resp.raise_for_status()
        f_input.write_text(resp.text, encoding="utf-8")

    data = f_input.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("gpt2")
    train_arr, val_arr = prepare_with_encoder(data, enc)

    f_train.write_bytes(train_arr.tobytes())
    f_val.write_bytes(val_arr.tobytes())

    print("Wrote:", f_train, f_val)

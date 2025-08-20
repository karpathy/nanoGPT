from __future__ import annotations

from typing import Tuple, Protocol
from pathlib import Path
import pickle
import numpy as np
import tiktoken

from ml_playground.experiments import register


class Encoder(Protocol):
    def encode_ordinary(self, text: str) -> list[int]: ...


def prepare_with_encoder(
    text: str, enc: Encoder
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Split 90/10 and encode with provided encoder; return arrays and meta.

    Arrays are dtype uint32 to accommodate large token IDs.
    Meta contains encoding name (if available) and dtype.
    """
    n = len(text)
    train_data = text[: int(n * 0.9)]
    val_data = text[int(n * 0.9) :]

    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    train_arr = np.array(train_ids, dtype=np.uint32)
    val_arr = np.array(val_ids, dtype=np.uint32)

    # Best-effort to get name
    enc_name = getattr(enc, "name", None) or getattr(enc, "_name", None) or "unknown"
    meta = {"encoding": enc_name, "dtype": "uint32"}
    return train_arr, val_arr, meta


def prepare_from_text(
    text: str, encoding_name: str = "cl100k_base"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    enc = tiktoken.get_encoding(encoding_name)
    return prepare_with_encoder(text, enc)


@register("bundestag_tiktoken")
def main() -> None:
    """Prepare a BPE-tokenized dataset for experiments/bundestag_tiktoken.

    - Reads input.txt (prefers experiments/bundestag_tiktoken/datasets/input.txt, with fallback to bundled resource)
    - Splits 90/10 into train/val
    - Encodes using tiktoken BPE and writes uint32 arrays
    - Writes train.bin, val.bin, and meta.pkl under experiments/bundestag_tiktoken/datasets
    """
    ds_dir = Path("ml_playground") / "experiments" / "bundestag_tiktoken" / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "input.txt"

    if not input_file_path.exists():
        exp_candidates = [
            Path("ml_playground")
            / "experiments"
            / "bundestag_tiktoken"
            / "datasets"
            / "input.txt",
            Path("ml_playground") / "experiments" / "bundestag_tiktoken" / "input.txt",
        ]
        src: Path | None = None
        for p in exp_candidates:
            if p.exists():
                src = p
                break
        if src is None:
            bundled = Path(__file__).parent / "input.txt"
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

    enc_name = "cl100k_base"
    enc = tiktoken.get_encoding(enc_name)

    train_arr, val_arr, meta = prepare_with_encoder(data, enc)

    (ds_dir / "train.bin").write_bytes(train_arr.tobytes())
    (ds_dir / "val.bin").write_bytes(val_arr.tobytes())

    # Persist tokenizer metadata so sampling matches training.
    with (ds_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    print("Wrote:", ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl")

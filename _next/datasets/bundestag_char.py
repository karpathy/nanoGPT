from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
from . import register


@register("bundestag_char")
def main() -> None:
    """Prepare a character-level dataset from data/bundestag_char/page1.txt

    - Reads page1.txt from data/bundestag_char
    - Splits 90/10 into train/val
    - Builds char-level vocab and encodes to uint16 ids
    - Writes train.bin, val.bin, and meta.pkl (stoi/itos/vocab_size)
    """
    ds_dir = Path("data") / "bundestag_char"
    ds_dir.mkdir(parents=True, exist_ok=True)
    input_file_path = ds_dir / "page1.txt"

    if not input_file_path.exists():
        # Auto-seed from bundled resource if the input file is missing
        bundled = Path(__file__).parent / "resources" / "bundestag_char" / "page1.txt"
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

    chars = sorted(set(data))
    vocab_size = len(chars)
    print("vocab size:", vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    (ds_dir / "train.bin").write_bytes(train_ids.tobytes())
    (ds_dir / "val.bin").write_bytes(val_ids.tobytes())

    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    with (ds_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    print("Wrote:", ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl")

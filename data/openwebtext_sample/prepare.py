# Save a *sample* of OpenWebText to binary files for training.
#
# This is intended for real (non-toy) sanity runs on a single GPU without downloading
# the full ~54GB OpenWebText HF cache used by data/openwebtext/prepare.py.
#
# Example:
#   python data/openwebtext_sample/prepare.py --subset 'train[:1%]' --val_frac 0.01
#
# Outputs:
#   data/openwebtext_sample/train.bin
#   data/openwebtext_sample/val.bin
#
# Notes:
# - Uses GPT-2 BPE via tiktoken, same as the full OpenWebText prep script.
# - The exact contents are determined by the HF dataset subset string and seed.

from __future__ import annotations

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="train[:1%]", help="HF split selector (e.g. 'train[:1%]' or 'train[:100000]')")
    parser.add_argument("--val_frac", type=float, default=0.01, help="fraction of the subset reserved for val")
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--num_proc", type=int, default=8, help="num_proc for dataset map/tokenization")
    parser.add_argument("--total_batches", type=int, default=128, help="shards used when writing memmap (higher = lower peak RAM)")
    args = parser.parse_args()

    if not (0.0 < args.val_frac < 1.0):
        raise ValueError("--val_frac must be in (0,1)")
    if args.total_batches <= 0:
        raise ValueError("--total_batches must be >= 1")

    enc = tiktoken.get_encoding("gpt2")

    # Download only the requested subset.
    # Note: OpenWebText only has a 'train' split; we create val from this subset.
    dset = load_dataset("openwebtext", split=args.subset, num_proc=args.num_proc)
    split_dataset = dset.train_test_split(test_size=args.val_frac, seed=args.seed, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=args.num_proc,
    )

    out_dir = os.path.dirname(__file__)
    for split, dset_tok in tokenized.items():
        arr_len = np.sum(dset_tok["len"], dtype=np.uint64)
        filename = os.path.join(out_dir, f"{split}.bin")
        dtype = np.uint16  # gpt2 vocab < 2**16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        idx = 0
        for batch_idx in tqdm(range(args.total_batches), desc=f"writing {filename}"):
            # Avoid `with_format("numpy")` here: HF datasets + NumPy>=2.0 can error when it
            # tries to create zero-copy object arrays. The default Python lists work fine.
            batch = dset_tok.shard(num_shards=args.total_batches, index=batch_idx, contiguous=True)
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Quick metadata
    train_tokens = os.path.getsize(os.path.join(out_dir, "train.bin")) // 2
    val_tokens = os.path.getsize(os.path.join(out_dir, "val.bin")) // 2
    print(f"wrote data/openwebtext_sample/train.bin (~{train_tokens:,} tokens)")
    print(f"wrote data/openwebtext_sample/val.bin   (~{val_tokens:,} tokens)")


if __name__ == "__main__":
    main()

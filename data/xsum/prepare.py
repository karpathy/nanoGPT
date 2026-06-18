import os
import pickle

import tiktoken
import torch
from datasets import load_dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_FILES = {
    "train": "train.pt",
    "val": "val.pt",
    "test": "test.pt",
}

MAX_LENGTH = 1024
IGNORE_INDEX = -1
TRAIN_LIMIT = 50000
VAL_LIMIT = 5000
TEST_LIMIT = 5000
SEED = 1337

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def encode_text(text):
    return enc.encode_ordinary(text)


def build_example(document, summary):
    prompt_text = f"Document:\n{document.strip()}\n\nSummary:\n"
    target_text = summary.strip() + "\n"

    prompt_ids = encode_text(prompt_text)
    target_ids = encode_text(target_text)

    max_prompt_len = MAX_LENGTH - len(target_ids) - 1
    if max_prompt_len <= 0:
        return None

    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[:max_prompt_len]

    input_ids = prompt_ids + target_ids + [eot]
    labels = ([IGNORE_INDEX] * len(prompt_ids)) + target_ids + [eot]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "gold_summary": summary.strip(),
    }


def process_split(dataset_split, limit=None):
    if limit is not None and len(dataset_split) > limit:
        dataset_split = dataset_split.shuffle(seed=SEED).select(range(limit))

    rows = []
    dropped = 0

    for row in dataset_split:
        example = build_example(row["document"], row["summary"])
        if example is None:
            dropped += 1
            continue
        rows.append(example)

    return rows, dropped


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset = load_dataset("EdinburghNLP/xsum")

    splits = {
        "train": (dataset["train"], TRAIN_LIMIT),
        "val": (dataset["validation"], VAL_LIMIT),
        "test": (dataset["test"], TEST_LIMIT),
    }

    stats = {}

    for split_name, (split_dataset, limit) in splits.items():
        rows, dropped = process_split(split_dataset, limit=limit)
        out_path = os.path.join(DATA_DIR, OUT_FILES[split_name])
        torch.save(rows, out_path)

        stats[split_name] = {
            "num_examples": len(rows),
            "dropped": dropped,
            "limit": limit,
        }

        print(
            f"{split_name}: saved {len(rows):,} examples "
            f"(dropped {dropped:,}) -> {out_path}"
        )

    meta = {
        "vocab_size": 50257,
        "task": "xsum",
        "format": "supervised_causal_lm_summarization",
        "ignore_index": IGNORE_INDEX,
        "tokenizer": "gpt2",
        "max_length": MAX_LENGTH,
        "splits": stats,
    }

    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\nDone. Files:")
    for fname in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, fname)
        print(f"  {fname}: {os.path.getsize(path) / 1e6:.2f} MB")
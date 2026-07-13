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
VAL_SIZE = 1000
MAX_LENGTH = 1024
IGNORE_INDEX = -1

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def encode_text(text):
    return enc.encode_ordinary(text)

def build_example(question, answer):
    prompt_text = f"Question: {question.strip()}\nAnswer:\n"
    answer_text = answer.strip()
    gold_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text

    prompt_ids = encode_text(prompt_text)
    target_ids = encode_text(answer_text + "\n")

    input_ids = prompt_ids + target_ids + [eot]
    labels = ([IGNORE_INDEX] * len(prompt_ids)) + target_ids + [eot]

    if len(input_ids) > MAX_LENGTH:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "gold_answer": gold_answer,
    }

def process_split(dataset_split):
    rows = []
    dropped = 0
    for row in dataset_split:
        example = build_example(row["question"], row["answer"])
        if example is None:
            dropped += 1
            continue
        rows.append(example)
    return rows, dropped

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset = load_dataset("openai/gsm8k", "main")
    train_valid = dataset["train"].train_test_split(
        test_size=VAL_SIZE,
        seed=1337,
        shuffle=True,
    )
    splits = {
        "train": train_valid["train"],
        "val": train_valid["test"],
        "test": dataset["test"],
    }

    stats = {}
    for split_name, split_dataset in splits.items():
        rows, dropped = process_split(split_dataset)
        out_path = os.path.join(DATA_DIR, OUT_FILES[split_name])
        torch.save(rows, out_path)
        stats[split_name] = {
            "num_examples": len(rows),
            "dropped_over_length": dropped,
        }
        print(
            f"{split_name}: saved {len(rows):,} examples "
            f"(dropped {dropped:,} over MAX_LENGTH={MAX_LENGTH}) -> {out_path}"
        )

    meta = {
        "vocab_size": 50257,
        "task": "gsm8k",
        "format": "supervised_causal_lm",
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
import os
import pickle
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-t", "--token_file", default=None)
    return parser.parse_args()


def process_data(input_file, token_file):
    with open(input_file, "r") as f:
        data = f.read()

    if token_file != None:
        with open(token_file, "r") as f:
            token_data = f.read()
    else:
        token_data = data

    print(f"Length of dataset: {len(data):,}")

    chars = sorted(list(set(token_data)))
    vocab_size = len(chars)

    print(f"Unique chars: {''.join(chars)}")
    print(f"Vocab size: {vocab_size:,}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    train_ids = encode(train_data)
    val_ids = encode(val_data)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    return train_ids, val_ids, stoi, itos


def save_data(train_ids, val_ids, stoi, itos, output_dir):
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))

    meta = {"vocab_size": len(stoi), "itos": itos, "stoi": stoi}
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    args = parse_args()
    train_ids, val_ids, stoi, itos = process_data(args.input_file, args.token_file)
    save_data(train_ids, val_ids, stoi, itos, ".")

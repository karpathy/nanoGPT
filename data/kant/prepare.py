# data/kant/prepare.py
import os
import pickle
import numpy as np

# Path to input file
input_file_path = os.path.join(os.path.dirname(__file__), 'data1.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"length of dataset in characters: {len(data):,}")

# Unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size}")

# Mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Train/val split
n = len(data)
train_data = data[:int(n * 0.9)]
val_data   = data[int(n * 0.9):]

# Encode to integers
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids   = np.array(encode(val_data), dtype=np.uint16)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Save directly in data/kant/
out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, 'train.bin'))
val_ids.tofile(os.path.join(out_dir, 'val.bin'))

# Save meta info
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")
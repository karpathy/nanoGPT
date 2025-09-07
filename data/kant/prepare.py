# data/kant/prepare.py
import os
import pickle
import requests
import numpy as np

# Read your uploaded file
input_file_path = os.path.join(os.path.dirname(__file__), 'data1.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"length of dataset in characters: {len(data):,}")

# Get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size}")

# Mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Train/val split
n = len(data)
train_data = data[:int(n*0.9)]
val_data   = data[int(n*0.9):]

# Encode to integers
train_ids = encode(train_data)
val_ids   = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
os.makedirs(os.path.join(os.path.dirname(__file__), 'processed'), exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids   = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'processed/train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'processed/val.bin'))

# Save meta information
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'processed/meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")

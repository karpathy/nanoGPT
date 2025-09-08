# data/kant/prepare.py
import os
import requests
import tiktoken
import numpy as np
import pickle

# Path to dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'data1.txt')

# Download if not exists (replace URL if you want other text)
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# Read dataset
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"length of dataset in characters: {len(data):,}")

# Train/val split
n = len(data)
train_data = data[:int(n * 0.9)]
val_data   = data[int(n * 0.9):]

# Encode with GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids   = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Convert to numpy arrays
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids   = np.array(val_ids, dtype=np.uint16)

# Save binary files
out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, 'train.bin'))
val_ids.tofile(os.path.join(out_dir, 'val.bin'))

# Save meta info (encoder/decoder)
meta = {
    'vocab_size': enc.n_vocab,
    'encoder': enc,  # not JSON/pickle safe fully, but reference
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")

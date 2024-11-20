import os
import numpy as np
from datasets import load_dataset  # huggingface datasets
import pickle

# Load enwik8 dataset
dataset = load_dataset("enwik8")

# Extract the full text
raw_text = ''.join(dataset["train"][:]['text'])  # Combine all text
#text = strip_xml_tags(raw_text)       # Remove XML tags
text = raw_text

# Split text into train (90M), test (5M), and val (5M)
train_data = text[:90_000_000]
val_data = text[90_000_000:95_000_000]
test_data = text[95_000_000:100_000_000]

# get all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train, validation, and test splits
#n = len(text)

# encode all splits to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
import os
import requests
import tiktoken
import numpy as np
import re


def preprocess_text(text):
    # Normalize the text to lowercase
    text = text.lower()
    
    # Use regex to find words, ignoring punctuation
    words = re.findall(r'\b\w+\b', text)
    
    return words


def preprocess_text(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b|:', text)
    return words

def find_unique_words(text):
    words = preprocess_text(text)
    
    # Use a set to collect unique words
    unique_words = set(words)
    
    return unique_words


# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# data = preprocess_text(data)
# data = ' '.join(data) # concatenate the words back together
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_synth.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_synth.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens

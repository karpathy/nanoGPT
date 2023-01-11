import os
import requests
import tiktoken
import numpy as np

class Dataset:
    def __init__(self, data_url = None) -> None:
        self.data_url = data_url
        self.input_data = None
        self.train_ids = None
        self.val_ids = None

    def fetch(self):
        data_url = self.data_url or 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' # shakespeare example if not set
        self.input_data = requests.get(data_url, timeout=1024).text

    def save(self, path):
        with open(path, 'w', encoding="utf-8") as f:
            f.write(self.input_data)
    
    def load(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            self.input_data = f.read()

    def parse(self):
        data = self.input_data
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode with tiktoken gpt2 bpe
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        print(f"train has {len(train_ids)} tokens")
        print(f"val has {len(val_ids)} tokens")

        # export to bin files
        self.train_ids = np.array(train_ids, dtype=np.uint16)
        self.val_ids = np.array(val_ids, dtype=np.uint16)
    
    def export(self, path, name):
        self.train_ids.tofile(path + name + '_train.bin')
        self.val_ids.tofile(path + name + '_val.bin')
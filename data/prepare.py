import os
import pickle
import sys
import numpy as np  # add this import at the top

def prepare(config):
    input_file = os.path.join('data', config.dataset, 'input.txt')
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("Vocabulary size:", vocab_size)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])

    n = len(data)
    train_data = encode(data[:int(n*0.9)])
    val_data = encode(data[int(n*0.9):])

    os.makedirs(f"data/{config.dataset}", exist_ok=True)
    with open(f"data/{config.dataset}/train.bin", 'wb') as f:
        np.array(train_data, dtype=np.uint16).tofile(f)
    with open(f"data/{config.dataset}/val.bin", 'wb') as f:
        np.array(val_data, dtype=np.uint16).tofile(f)
    with open(f"data/{config.dataset}/meta.pkl", 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)

    print("✅ Saved train.bin, val.bin, meta.pkl")

if __name__ == "__main__":
    config_path = sys.argv[1].split("=")[1]
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    prepare(config)

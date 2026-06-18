import os, urllib.request, pickle
import numpy as np
import tiktoken
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

HF_API = "https://datasets-server.huggingface.co/parquet?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1"

import urllib.request, json

with urllib.request.urlopen(HF_API) as r:
    parquet_info = json.loads(r.read())

# build {split: [url, ...]} mapping
split_urls = {}
for entry in parquet_info["parquet_files"]:
    split = entry["split"]
    split_urls.setdefault(split, []).append(entry["url"])

print(f"Found splits: {list(split_urls.keys())}")

# ── tokenise ─────
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

SPLITS_OUT = {
    "train":      "train.bin",
    "validation": "val.bin",
}

for split, out_fname in SPLITS_OUT.items():
    urls   = split_urls[split]
    tokens = []

    for i, url in enumerate(urls):
        local_pq = f"/tmp/wikitext2_{split}_{i}.parquet"
        print(f"  Downloading {split} shard {i+1}/{len(urls)}...")
        urllib.request.urlretrieve(url, local_pq)

        df = pd.read_parquet(local_pq)
        for text in df['text']:
            if isinstance(text, str) and text.strip():
                ids = enc.encode_ordinary(text)
                if ids:
                    ids.append(eot)
                    tokens.extend(ids)

    out_path = os.path.join(DATA_DIR, out_fname)
    np.array(tokens, dtype=np.uint16).tofile(out_path)
    print(f"  {split}: {len(tokens):,} tokens → {out_path}")

# ── meta ─────
with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": 50257}, f)

print(f"\nDone. Files:")
for fname in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, fname)
    print(f"  {fname}: {os.path.getsize(path)/1e6:.2f} MB")
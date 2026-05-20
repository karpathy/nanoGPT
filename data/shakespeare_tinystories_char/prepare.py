"""
Prepare a combined Shakespeare + TinyStories char-level dataset.

Same shape as shakespeare_wiki_char/prepare.py but the second corpus is
roneneldan/TinyStories (a synthetic children's-stories dataset Karpathy used
for nanoGPT demos). TinyStories is much easier to learn at char-level than
wiki — both corpora are continuous narrative text, so the slider interpolates
between *two narrative styles* rather than between narrative and encyclopedia.

Pipeline:
  1. Download tinyshakespeare from karpathy/char-rnn.
  2. Download the first ~1.5 MB of TinyStories-train.txt from HuggingFace.
  3. Normalize shakespeare to match TinyStories's surface form: lowercase +
     space-padded punctuation, preserving newlines so play-script structure
     (speaker tag, colon, line break) survives.
  4. Concatenate (shakespeare, separator, tinystories) and build a single
     char vocab.
  5. 90/10 train/val split.
  6. Save train.bin, val.bin (uint16) and meta.pkl, identical layout to
     nanoGPT's shakespeare_char/prepare.py.
"""
import os
import re
import pickle
import urllib.request
import numpy as np

HERE = os.path.dirname(__file__)

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStories-train.txt"
)
TINYSTORIES_BYTES = 1_500_000  # ~1.5 MB, roughly shakespeare-equivalent volume
SEPARATOR = "\n\n===\n\n"


def _download(url, dst, range_bytes=None):
    if os.path.exists(dst):
        return
    req = urllib.request.Request(url)
    if range_bytes is not None:
        req.add_header("Range", f"bytes=0-{range_bytes - 1}")
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        f.write(r.read())


# --- 1. Shakespeare -----------------------------------------------------------
shake_raw_path = os.path.join(HERE, "shakespeare.txt")
_download(SHAKESPEARE_URL, shake_raw_path)
with open(shake_raw_path, "r", encoding="utf-8") as f:
    shake_raw = f.read()

# --- 2. TinyStories slice -----------------------------------------------------
ts_path = os.path.join(HERE, "tinystories.txt")
_download(TINYSTORIES_URL, ts_path, range_bytes=TINYSTORIES_BYTES)
with open(ts_path, "r", encoding="utf-8") as f:
    tinystories = f.read()

# --- 3. Normalize shakespeare to TinyStories surface form ---------------------
# TinyStories is already lowercased, simple-vocab children's prose with
# space-separated punctuation. Apply the same convention to shakespeare so
# the vocabularies merge cleanly into one small char set.
_PUNCT_CHARS = r"""!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"""


def normalize_shakespeare(text: str) -> str:
    text = text.lower()
    text = re.sub(f"([{_PUNCT_CHARS}])", r" \1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text


shake = normalize_shakespeare(shake_raw)

# --- 4. Concatenate -----------------------------------------------------------
# Drop non-printable-ASCII glyphs from TinyStories (occasional curly quotes,
# em-dashes etc) to keep the char vocab tight (~60-80) so the model and
# embedding table stay Colab-T4-sized.
_ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n"}


def ascii_filter(text: str) -> str:
    return "".join(c for c in text if c in _ALLOWED)


shake = ascii_filter(shake)
tinystories = ascii_filter(tinystories)

data = shake + SEPARATOR + tinystories
print(f"shakespeare chars : {len(shake):,}")
print(f"tinystories chars : {len(tinystories):,}")
print(f"combined chars    : {len(data):,}")

# --- Vocab --------------------------------------------------------------------
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


# --- 5. Split -----------------------------------------------------------------
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# --- 6. Save ------------------------------------------------------------------
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(HERE, "train.bin"))
val_ids.tofile(os.path.join(HERE, "val.bin"))

meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

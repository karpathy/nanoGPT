"""
Prepare a combined Shakespeare + Wikipedia (wiki-abc 100MB tier, first ~1.5 MB)
dataset for character-level language modeling.

Pipeline:
  1. Download tinyshakespeare from karpathy/char-rnn.
  2. Download the first ~1.5 MB of wiki-abc/wiki_100MB.txt via HTTP range so
     neither corpus dominates the char vocab.
  3. Normalize shakespeare to match wiki-abc's surface form: lowercase +
     space-padded punctuation, while preserving newlines so play-script
     structure (speaker tag, colon, line break) survives.
  4. Concatenate (shakespeare, separator, wiki) and build a single char vocab.
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
WIKI_URL = (
    "https://storage.googleapis.com/"
    "abc-w2v-runs-praxis-tractor-469320-g8/data/wiki_100MB.txt"
)
WIKI_BYTES = 1_500_000  # ~1.5 MB, roughly shakespeare-equivalent volume
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

# --- 2. Wiki slice ------------------------------------------------------------
wiki_path = os.path.join(HERE, "wiki.txt")
_download(WIKI_URL, wiki_path, range_bytes=WIKI_BYTES)
with open(wiki_path, "r", encoding="utf-8") as f:
    wiki = f.read()

# --- 3. Normalize shakespeare to wiki surface form ----------------------------
# wiki-abc style: lowercased, punctuation tokens space-padded, words separated
# by single spaces. We want the *same* surface form but keep '\n' so the
# play-script line structure (e.g. "First Citizen:\n...") survives.
_PUNCT_CHARS = r"""!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"""


def normalize_shakespeare(text: str) -> str:
    text = text.lower()
    # space-pad every punctuation char, preserving newlines verbatim
    text = re.sub(f"([{_PUNCT_CHARS}])", r" \1 ", text)
    # collapse runs of spaces/tabs but not newlines
    text = re.sub(r"[ \t]+", " ", text)
    # trim trailing spaces before newlines and leading spaces after
    text = re.sub(r" *\n *", "\n", text)
    return text


shake = normalize_shakespeare(shake_raw)

# --- 4. Concatenate -----------------------------------------------------------
# Drop non-printable-ASCII glyphs that leak in from wiki-abc biographical
# entries (CJK, accented latin, etc). This keeps the char vocab tight (~60-80)
# so the model and embedding table stay Colab-T4-sized.
_ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n"}


def ascii_filter(text: str) -> str:
    return "".join(c for c in text if c in _ALLOWED)


shake = ascii_filter(shake)
wiki = ascii_filter(wiki)

data = shake + SEPARATOR + wiki
print(f"shakespeare chars: {len(shake):,}")
print(f"wiki chars       : {len(wiki):,}")
print(f"combined chars   : {len(data):,}")

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

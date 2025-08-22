import os
import pickle
from .base import BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def train(self, corpus_file, vocab_size=None):
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = f.read()

        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])

    def save(self, path):
        meta = {
            'vocab_size': self.vocab_size,
            'itos': self.itos,
            'stoi': self.stoi,
        }
        with open(path, 'wb') as f:
            pickle.dump(meta, f)

    def load(self, path):
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        self.itos = meta['itos']
        self.stoi = meta['stoi']
        self.chars = list(self.stoi.keys())

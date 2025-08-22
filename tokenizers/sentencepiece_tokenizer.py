import os
import sentencepiece as spm
from .base import BaseTokenizer

class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = 0

    def train(self, corpus_file, vocab_size=8000, model_type='bpe', model_prefix='tokenizer'):
        self.vocab_size = vocab_size
        command = (
            f'--input={corpus_file} --model_prefix={model_prefix} '
            f'--vocab_size={vocab_size} --model_type={model_type}'
        )
        spm.SentencePieceTrainer.train(command)
        self.load(f'{model_prefix}.model')

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def save(self, path):
        # The model is already saved during training, this method is for interface consistency
        # We can copy the model file to the specified path if needed
        model_file = self.sp.model_proto()
        with open(path, 'wb') as f:
            f.write(model_file)

    def load(self, model_path):
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()

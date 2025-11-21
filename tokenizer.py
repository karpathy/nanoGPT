import abc
from abc import abstractmethod

import tiktoken


class Tokenizer(abc.ABC):

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass


class TiktokenTokenizer(Tokenizer):

    def __init__(self, encoding: tiktoken.Encoding, allowed_special: set[str]):
        super().__init__()
        self._encoding = encoding
        self._allowed_special = allowed_special

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special=self._allowed_special)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    @staticmethod
    def gpt2_tokenizer():
        return TiktokenTokenizer(tiktoken.get_encoding("gpt2"), allowed_special={"<|endoftext|>"})


class DictBasedTokenizer(Tokenizer):

    def __init__(self, stoi: dict[str,int], itos: dict[int,str]):
        super().__init__()
        self._stoi = stoi
        self._itos = itos

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self._itos[i] for i in tokens])

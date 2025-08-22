from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, ids):
        pass

    @abstractmethod
    def train(self, corpus_file):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

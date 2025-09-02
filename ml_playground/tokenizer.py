from __future__ import annotations

from typing import Protocol, List, Dict, Tuple, Callable, Optional
from pathlib import Path
import abc


class Tokenizer(Protocol):
    """Protocol for tokenizers that can encode/decode text to/from token IDs."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        ...
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        ...
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of this tokenizer."""
        ...
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping (token -> ID)."""
        ...


class CharTokenizer:
    """Character-level tokenizer."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.stoi = vocab
            self.itos = {i: s for s, i in vocab.items()}
        else:
            # Default character-level vocabulary will be built during training
            self.stoi = {}
            self.itos = {}
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]
    
    def decode(self, token_ids: List[int]) -> str:
        return ''.join([self.itos.get(i, '') for i in token_ids])
    
    def get_vocab_size(self) -> int:
        return len(self.stoi)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.stoi.copy()


class WordTokenizer:
    """Word-level tokenizer."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.stoi = vocab
            self.itos = {i: s for s, i in vocab.items()}
        else:
            # Default word-level vocabulary will be built during training
            self.stoi = {}
            self.itos = {}
    
    def encode(self, text: str) -> List[int]:
        import re
        # Simple word tokenization
        words = re.findall(r'\w+|[^\w\s]', text)
        return [self.stoi.get(word, 0) for word in words]
    
    def decode(self, token_ids: List[int]) -> str:
        return ' '.join([self.itos.get(i, '') for i in token_ids])
    
    def get_vocab_size(self) -> int:
        return len(self.stoi)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.stoi.copy()


class TiktokenTokenizer:
    """Tiktoken-based BPE tokenizer."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer but is not installed. "
                "Please install it with `pip install tiktoken`."
            )
        
        self.encoding_name = encoding_name
        self.encoder = tiktoken.get_encoding(encoding_name)
    
    def encode(self, text: str) -> List[int]:
        return self.encoder.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, token_ids: List[int]) -> str:
        return self.encoder.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        return self.encoder.n_vocab
    
    def get_vocab(self) -> Dict[str, int]:
        # Tiktoken doesn't expose vocab directly, but we can get the encoder mappings
        return dict(self.encoder._special_tokens)


def create_tokenizer(tokenizer_type: str, **kwargs) -> Tokenizer:
    """Factory function to create a tokenizer based on type."""
    if tokenizer_type == "char":
        return CharTokenizer(**kwargs)
    elif tokenizer_type == "word":
        return WordTokenizer(**kwargs)
    elif tokenizer_type == "tiktoken":
        encoding_name = kwargs.get("encoding_name", "cl100k_base")
        return TiktokenTokenizer(encoding_name=encoding_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

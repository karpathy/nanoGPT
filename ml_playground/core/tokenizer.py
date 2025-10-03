from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Literal
from types import MappingProxyType
from ml_playground.core.tokenizer_protocol import Tokenizer


__all__ = ["Tokenizer", "create_tokenizer"]


class CharTokenizer:
    """Character-level tokenizer that maps single characters to integer ids."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self._name = "char"
        if vocab is not None:
            self.stoi = vocab
            self.itos = {i: s for s, i in vocab.items()}
        else:
            # Default character-level vocabulary will be built during training
            self.stoi = {}
            self.itos = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def vocab(self) -> Mapping[str, int]:
        return MappingProxyType(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in token_ids)


class WordTokenizer:
    """Word-level tokenizer that segments text via a simple regex pattern."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self._name = "word"
        if vocab is not None:
            self.stoi = vocab
            self.itos = {i: s for s, i in vocab.items()}
        else:
            # Default word-level vocabulary will be built during training
            self.stoi = {}
            self.itos = {}

    @property
    def name(self) -> str:
        return self._name

    def encode(self, text: str) -> list[int]:
        import re

        # Simple word tokenization
        words = re.findall(r"\w+|[^\w\s]", text)
        return [self.stoi.get(word, 0) for word in words]

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self.itos.get(int(i), "") for i in token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def vocab(self) -> Mapping[str, int]:
        return MappingProxyType(self.stoi)


class TiktokenTokenizer:
    """`tiktoken`-based BPE tokenizer supporting GPT-style byte pair encoding."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        *,
        loader: Callable[[], Any] | None = None,
    ):
        module_loader = loader if loader is not None else lambda: __import__("tiktoken")
        try:
            tiktoken_module = module_loader()
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer but is not installed. "
                "Please install it with `pip install tiktoken`."
            ) from exc

        self.encoding_name = encoding_name
        self.encoder = tiktoken_module.get_encoding(encoding_name)
        self._name = "tiktoken"

    @property
    def name(self) -> str:
        return self._name

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.encoder.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.encoder.n_vocab

    @property
    def vocab(self) -> Mapping[str, int]:
        # tiktoken exposes mergeable ranks as a dict[str, int]; use it when available
        ranks = getattr(self.encoder, "_mergeable_ranks", None)
        if isinstance(ranks, dict):
            return MappingProxyType(ranks)
        # Fallback to empty mapping if ranks is not available or not a dict
        return MappingProxyType({})


def create_tokenizer(
    tokenizer_type: Literal["char", "word", "tiktoken"], **kwargs
) -> Tokenizer:
    """Factory for known tokenizer implementations.

    Args:
        tokenizer_type: Name of the tokenizer family to instantiate.
        **kwargs: Implementation-specific keyword arguments (e.g., vocab, encoding_name).

    Returns:
        A concrete `Tokenizer` implementation associated with the supplied name.

    Raises:
        ValueError: If an unknown tokenizer type is requested.
    """
    if tokenizer_type == "char":
        return CharTokenizer(**kwargs)
    if tokenizer_type == "word":
        return WordTokenizer(**kwargs)
    if tokenizer_type == "tiktoken":
        encoding_name = kwargs.pop("encoding_name", "cl100k_base")
        loader = kwargs.pop("loader", None)
        return TiktokenTokenizer(encoding_name=encoding_name, loader=loader)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

from __future__ import annotations

from typing import Protocol, List, Mapping


__all__ = ["Tokenizer"]


class Tokenizer(Protocol):
    """Protocol for tokenizers that can encode/decode text to/from token IDs."""

    # A human-readable tokenizer identifier (e.g., "char", "word", "tiktoken")
    @property
    def name(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def vocab(self) -> Mapping[str, int]: ...

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        ...

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        ...

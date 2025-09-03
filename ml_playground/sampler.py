"""ml_playground.sampler: sampling utilities.

Device seeding/TF32 is centrally handled in the CLI. This module constructs
device, dtype, and autocast contexts locally without exposing legacy shims.
"""

from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Tuple, Protocol, Optional
import logging
import pickle
import torch
from torch import autocast

from ml_playground.checkpoint import Checkpoint, CheckpointManager
from ml_playground.config import ModelConfig, SamplerConfig
from ml_playground.error_handling import CheckpointError, DataError, setup_logging
from ml_playground.model import GPT
from ml_playground.prepare import setup_tokenizer
from ml_playground.tokenizer import (
    CharTokenizer,
    WordTokenizer,
    TiktokenTokenizer,
    Tokenizer,
)


"""
Centralized sampling utilities for ml_playground experiments.

This module provides standardized utilities for model sampling including:
- Checkpoint loading with proper error handling
- Codec management for different tokenizer types
- Standardized encode/decode operations
- Error handling with centralized exception types

All experiments should use these utilities to ensure consistency and proper error handling.
"""


class Sampler(Protocol):
    def __call__(self, cfg: SamplerConfig) -> None: ...


def _load_checkpoint(
    out_dir: Path,
    device: str,
    logger: logging.Logger,
    use_best: bool,
) -> Checkpoint:
    """Load model checkpoint.

    Strict: surface errors to caller.
    """
    ckpt_mgr = CheckpointManager(out_dir=out_dir)
    if use_best:
        return ckpt_mgr.load_best_checkpoint(device=device, logger=logger)
    # Prefer latest; if none found, fall back to best, then raise
    try:
        return ckpt_mgr.load_latest_checkpoint(device=device, logger=logger)
    except CheckpointError as e_latest:
        # try best as a secondary rotated source
        return ckpt_mgr.load_best_checkpoint(device=device, logger=logger)


    # No stable-file fallback – strict mode requires rotated checkpoints.


def sample(cfg: SamplerConfig) -> None:
    """Sample from a trained model."""
    # --- Setup -------------------------------------------------------------------
    runtime_cfg = cfg.runtime
    sample_cfg = cfg.sample
    if not runtime_cfg:
        raise ValueError("Runtime configuration is missing")

    setup_logging(str(runtime_cfg.out_dir))
    logger = logging.getLogger(__name__)

    # --- Set random seeds -------------------------------------------------------
    torch.manual_seed(runtime_cfg.seed)
    torch.cuda.manual_seed(runtime_cfg.seed)

    # --- Device setup -----------------------------------------------------------
    device_type = "cuda" if "cuda" in runtime_cfg.device else "cpu"
    pt_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[runtime_cfg.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else autocast(device_type=device_type, dtype=pt_dtype)
    )

    # --- Load checkpoint --------------------------------------------------------
    checkpoint = _load_checkpoint(
        runtime_cfg.out_dir,
        runtime_cfg.device,
        logger,
        use_best=getattr(sample_cfg, "use_best_checkpoint", False),
    )

    # --- Model setup ------------------------------------------------------------
    model_cfg = ModelConfig(**checkpoint.model_args)
    model = GPT(model_cfg)
    model.load_state_dict(checkpoint.model, strict=False)
    model.eval()
    model.to(runtime_cfg.device)
    if runtime_cfg.compile:
        model = torch.compile(model)  # type: ignore

    # --- Tokenizer setup --------------------------------------------------------
    tokenizer = setup_tokenizer(runtime_cfg.out_dir)
    if not tokenizer:
        raise DataError(
            f"Tokenizer metadata not found in {runtime_cfg.out_dir} (expected meta.pkl)."
        )

    # --- Sampling ---------------------------------------------------------------
    start_text = sample_cfg.start
    if isinstance(start_text, str) and start_text.startswith("FILE:"):
        prompt_path = Path(start_text[5:])
        try:
            start_text = prompt_path.read_text(encoding="utf-8")
        except Exception as e:  # pragma: no cover - robust IO guard
            logger.error(f"Failed to read prompt file {prompt_path}: {e}")
            return
    start_ids = tokenizer.encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=runtime_cfg.device)[None, ...]

    logger.info("Sampling...")
    with torch.no_grad():
        with ctx:
            for k in range(sample_cfg.num_samples):
                y = model.generate(  # type: ignore
                    x,
                    sample_cfg.max_new_tokens,
                    temperature=sample_cfg.temperature,
                    top_k=sample_cfg.top_k,
                )
                output = tokenizer.decode(y[0].tolist())
                logger.info(output)
                logger.info("---------------")


def create_codec_from_tokenizer_type(
    tokenizer_type: str, **kwargs
) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Create encode/decode callables based on tokenizer type.

    Args:
        tokenizer_type: Type of tokenizer ('char', 'word', 'tiktoken')
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        Tuple of (encode_func, decode_func)

    Raises:
        DataError: If tokenizer type is unsupported or dependencies are missing
    """
    try:
        if tokenizer_type == "char":
            # For char tokenizer, we need vocab
            vocab = kwargs.get("vocab")
            if vocab is None:
                raise DataError("Char tokenizer requires 'vocab' parameter")
            char_tokenizer: Tokenizer = CharTokenizer(vocab=vocab)
            return (char_tokenizer.encode, char_tokenizer.decode)
        elif tokenizer_type == "word":
            # For word tokenizer, we need to pass the vocab
            vocab = kwargs.get("vocab")
            if vocab is None:
                raise DataError("Word tokenizer requires 'vocab' parameter")
            word_tokenizer: Tokenizer = WordTokenizer(vocab=vocab)
            return (word_tokenizer.encode, word_tokenizer.decode)
        elif tokenizer_type == "tiktoken":
            # For tiktoken tokenizer, we can use encoding_name
            encoding_name = kwargs.get("encoding_name", "gpt2")
            tiktoken_tokenizer: Tokenizer = TiktokenTokenizer(
                encoding_name=encoding_name
            )
            return (tiktoken_tokenizer.encode, tiktoken_tokenizer.decode)
        else:
            raise DataError(f"Unsupported tokenizer type: {tokenizer_type}")
    except ImportError as e:
        raise DataError(
            f"Required dependency for {tokenizer_type} tokenizer is not installed: {e}"
        ) from e
    except Exception as e:
        raise DataError(f"Failed to create {tokenizer_type} tokenizer: {e}") from e


def validate_and_create_codec(
    meta_path: Path | None, tokenizer_type: str = "auto", **kwargs
) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Validate and create encode/decode callables with strict behavior.

    Strict policy:
    - tokenizer_type == 'auto' requires a valid meta.pkl path; no fallbacks.
    - explicit tokenizer types must provide required kwargs; no defaults inferred.
    """
    if tokenizer_type == "auto":
        return _codec_from_meta(meta_path)

    return create_codec_from_tokenizer_type(tokenizer_type, **kwargs)


class CodecManager:
    """A utility class for managing encode/decode operations with multiple tokenizer types."""

    def __init__(self, meta_path: Path | None = None):
        self.meta_path = meta_path
        self._encode_func: Optional[Callable[[str], list[int]]] = None
        self._decode_func: Optional[Callable[[list[int]], str]] = None
        self._tokenizer_type: Optional[str] = None

    def initialize_codec(self, tokenizer_type: str = "auto", **kwargs) -> None:
        """Initialize the codec with the specified tokenizer type."""
        self._encode_func, self._decode_func = validate_and_create_codec(
            self.meta_path, tokenizer_type, **kwargs
        )
        self._tokenizer_type = tokenizer_type

    def encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        if self._encode_func is None:
            raise DataError("Codec not initialized. Call initialize_codec() first.")
        return self._encode_func(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        if self._decode_func is None:
            raise DataError("Codec not initialized. Call initialize_codec() first.")
        return self._decode_func(tokens)

    @property
    def tokenizer_type(self) -> Optional[str]:
        """Get the current tokenizer type."""
        return self._tokenizer_type


def _codec_from_meta(
    meta_path: Path | None,
) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Derive encode/decode callables for sampling using tokenizer meta.

    Strict policy: meta_path must be provided and exist. No fallbacks to other tokenizers.
    """
    if meta_path is None or not meta_path.exists():
        raise DataError("meta.pkl is required to derive codec (no fallback allowed)")

    # Load meta from pickle file
    try:
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
    except Exception as e:
        raise DataError(f"Failed to load meta.pkl at {meta_path}: {e}") from e

    # Validate required fields
    mv = meta.get("meta_version")
    if mv is None:
        raise DataError("Missing required 'meta_version' in meta.pkl")

    kind = meta.get("kind")
    dtype = meta.get("dtype")
    if dtype not in {"uint16", "uint32"}:
        raise DataError(f"Unsupported or missing meta 'dtype': {dtype!r}")

    # Create appropriate tokenizer based on meta
    if kind == "char":
        stoi = meta["stoi"]
        char_codec: Tokenizer = CharTokenizer(vocab=stoi)
        return (char_codec.encode, char_codec.decode)
    elif kind == "tiktoken":
        enc_name = meta["encoding"]
        try:
            tiktoken_tokenizer_meta: Tokenizer = TiktokenTokenizer(
                encoding_name=enc_name
            )
            return (tiktoken_tokenizer_meta.encode, tiktoken_tokenizer_meta.decode)
        except ImportError as e:
            raise DataError(
                "tiktoken is required to decode with the provided meta (kind='tiktoken'). "
                "Install it or provide a char-level meta.pkl."
            ) from e
    elif kind == "word":
        stoi_meta = meta.get("stoi")
        if not isinstance(stoi_meta, dict):
            raise DataError("Invalid meta for word: expected stoi dict")
        word_tokenizer: Tokenizer = WordTokenizer(vocab=stoi_meta)
        return (word_tokenizer.encode, word_tokenizer.decode)
    else:
        raise DataError(f"Unsupported meta 'kind': {kind!r}")

from __future__ import annotations
from pathlib import Path
from typing import Callable, Tuple, Protocol, Optional
import pickle
import torch
from ml_playground.model import GPTConfig, GPT
from ml_playground.config import SamplerConfig, RuntimeConfig
from ml_playground.device import setup
from ml_playground.tokenizer import CharTokenizer, WordTokenizer, TiktokenTokenizer, Tokenizer
from ml_playground.error_handling import (
    CheckpointError,
    DataError,
    ModelError,
    setup_logging,
)
from ml_playground.trainer import Checkpoint

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


def load_checkpoint(
    out_dir: Path,
    device: str,
    best_name: str = "ckpt_best.pt",
    last_name: str = "ckpt_last.pt",
) -> Tuple[GPT, Checkpoint]:
    """Load a checkpoint and instantiate the corresponding model.

    Args:
        out_dir: Directory containing checkpoints
        device: Device to load model to
        best_name: Filename for best checkpoint
        last_name: Filename for last checkpoint

    Returns:
        Tuple of (model, checkpoint) where checkpoint is a Checkpoint object

    Raises:
        CheckpointError: If no checkpoint is found or loading fails
        ModelError: If model instantiation or state loading fails
    """
    candidates = [
        out_dir / best_name,
        out_dir / last_name,
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise CheckpointError(
            f"No checkpoint found in {out_dir} (tried: {tried}). "
            "Ensure training has produced checkpoints or configure RuntimeConfig filenames."
        )
    try:
        ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint at {ckpt_path}: {e}") from e
    if not isinstance(ckpt_dict, dict):
        raise CheckpointError(
            f"Checkpoint at {ckpt_path} is not a dict; got {type(ckpt_dict).__name__}. "
            "Expected a trainer-produced checkpoint."
        )
    required = {"model", "model_args"}
    missing = required - set(ckpt_dict.keys())
    if missing:
        raise CheckpointError(
            f"Checkpoint at {ckpt_path} missing required keys: {missing}"
        )
    model_args = extract_model_args_from_checkpoint(ckpt_dict)
    # Check that model_args contains required keys
    required_model_args = {"block_size"}
    missing_model_args = required_model_args - set(model_args.keys())
    if missing_model_args:
        raise CheckpointError(
            f"Checkpoint at {ckpt_path} missing required model_args keys: {missing_model_args}"
        )
    # Instantiate model from checkpoint
    try:
        model = GPT(GPTConfig(**model_args))
    except Exception as e:
        raise ModelError(
            f"Failed to instantiate model from checkpoint at {ckpt_path}: {e}"
        ) from e
    # Load state dict
    try:
        model.load_state_dict(ckpt_dict["model"])
    except Exception as e:
        raise ModelError(
            f"Failed to load model state_dict from checkpoint at {ckpt_path}: {e}"
        ) from e
    # Create Checkpoint object
    checkpoint = Checkpoint(
        model=ckpt_dict["model"],
        optimizer=ckpt_dict.get("optimizer", {}),
        model_args=model_args,
        iter_num=ckpt_dict.get("iter_num", 0),
        best_val_loss=ckpt_dict.get("best_val_loss", float('inf')),
        config=ckpt_dict.get("config", {}),
        ema=ckpt_dict.get("ema"),
    )
    return model, checkpoint


def extract_model_args_from_checkpoint(checkpoint: dict) -> dict:
    """Extract model arguments from a checkpoint, with backward compatibility."""
    if "model_args" in checkpoint:
        return checkpoint["model_args"]

    # Backward compatibility: try to extract from config
    if "config" in checkpoint and "model" in checkpoint["config"]:
        return checkpoint["config"]["model"]

    # If we can't find model args, return empty dict
    return {}


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
            tiktoken_tokenizer: Tokenizer = TiktokenTokenizer(encoding_name=encoding_name)
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
    """Validate and create encode/decode callables with flexible fallback options.

    Args:
        meta_path: Path to meta.pkl file
        tokenizer_type: Type of tokenizer ('auto', 'char', 'word', 'tiktoken')
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        Tuple of (encode_func, decode_func)
    """
    # If tokenizer_type is 'auto', try to infer from meta
    if tokenizer_type == "auto" and meta_path is not None and meta_path.exists():
        try:
            return _codec_from_meta(meta_path)
        except Exception:
            # Fall through to default handling
            pass

    # If we have a specific tokenizer type, use it
    if tokenizer_type != "auto":
        return create_codec_from_tokenizer_type(tokenizer_type, **kwargs)

    # Default fallback behavior (same as original _codec_from_meta)
    return _codec_from_meta(meta_path)


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
    """Derive encode/decode callables for sampling using the new tokenizer protocol."""
    if meta_path is None or not meta_path.exists():
        # Use tiktoken as default fallback
        try:
            tiktoken_codec: Tokenizer = TiktokenTokenizer(encoding_name="gpt2")
            return (tiktoken_codec.encode, tiktoken_codec.decode)
        except ImportError as e:
            raise DataError(
                "No usable dataset meta found. "
                "As a fallback we require 'tiktoken' to use GPT-2 BPE ('gpt2') encoding, but it is not installed. "
                "Install tiktoken or provide a valid meta.pkl."
            ) from e

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
            tiktoken_tokenizer_meta: Tokenizer = TiktokenTokenizer(encoding_name=enc_name)
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


def sample(exp: SamplerConfig) -> None:
    if exp.runtime is None:
        raise Exception(
            "SamplerConfig.runtime is not resolved; use load_experiment_toml or provide [sample.runtime]."
        )
    # Provide a sensible default logger if none was supplied in the config
    if exp.logger is None:
        logger = setup_logging("ml_playground.sample")
    else:
        logger = exp.logger
    rt = exp.runtime
    # Early progress: device/context
    logger.info("[sample] Initializing device and context...")
    device_type, ptdtype, ctx = setup(rt.device, rt.dtype, rt.seed)
    device = device_type  # for legacy compat
    logger.info(f"[sample] Using device: {device}")

    # Load model from checkpoint
    logger.info("[sample] Loading model from checkpoint...")
    try:
        model, ckpt = load_checkpoint(rt.out_dir, device)
    except Exception as e:
        logger.error(f"[sample] Failed to load checkpoint: {e}")
        raise

    model.eval()
    model.to(device)
    if rt.compile:
        model = torch.compile(model)  # type: ignore

    # Derive encode/decode functions from dataset metadata
    logger.info("[sample] Deriving codec from dataset metadata...")
    try:
        # Get meta_pkl path from the data config if available
        meta_pkl_path = None
        if hasattr(exp, "data") and exp.data and exp.data.meta_pkl:
            meta_pkl_path = exp.data.dataset_dir / exp.data.meta_pkl
        elif rt.out_dir:
            # Fallback to out_dir if data config is not available
            meta_pkl_path = rt.out_dir / "meta.pkl"
        codec_manager = CodecManager(meta_pkl_path)
        codec_manager.initialize_codec()
        encode, decode = codec_manager.encode, codec_manager.decode
    except Exception as e:
        logger.error(f"[sample] Failed to derive codec: {e}")
        raise

    # Sample loop
    start = exp.sample.start
    if start is None:
        start_ids = [encode("\n")[-1]]  # Default to newline
    elif start.startswith("FILE:"):
        # Read prompt from file
        prompt_path = Path(start[5:])
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        start_ids = encode(prompt_text)
        # Print separator for FILE: prompts
        print("---------------")
    else:
        start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Run generation
    logger.info(f"[sample] Generating {exp.sample.max_new_tokens} tokens...")
    with torch.no_grad():
        with ctx:
            # Use generate method if available
            if hasattr(model, "generate"):
                x = model.generate(
                    x,
                    exp.sample.max_new_tokens,
                    exp.sample.temperature,
                    exp.sample.top_k or 0,
                )
            else:
                for k in range(exp.sample.max_new_tokens):
                    if "model_args" not in ckpt or not isinstance(
                        ckpt["model_args"], dict
                    ):
                        raise CheckpointError("Checkpoint missing valid model_args")
                    model_args = ckpt["model_args"]
                    if "block_size" not in model_args:
                        raise CheckpointError(
                            "Checkpoint missing block_size in model_args"
                        )
                    block_size = model_args["block_size"]
                    if not isinstance(block_size, int):
                        raise CheckpointError(
                            f"block_size in checkpoint is not an integer: {type(block_size)}"
                        )
                    if x.size(1) >= block_size:
                        x = x[:, -block_size:]
                    logits, _ = model(x)
                    logits = logits[0, -1, :] / exp.sample.temperature
                    if exp.sample.top_k is not None and exp.sample.top_k > 0:
                        v, _ = torch.topk(
                            logits, min(exp.sample.top_k, logits.size(-1))
                        )
                        logits[logits < v[[-1]]] = -float("Inf")
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    # Ensure next_id has the same number of dimensions as x
                    if next_id.dim() == 1:
                        next_id = next_id.unsqueeze(1)
                    x = torch.cat((x, next_id), dim=1)

    # Decode and print result
    output = decode(x[0].tolist())
    print(output)
    logger.info("[sample] Generation complete.")

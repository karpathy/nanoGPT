from __future__ import annotations

from typing import Protocol, Any, Iterable
from pathlib import Path
import pickle
import numpy as np
from ml_playground.tokenizer_protocol import Tokenizer
from ml_playground.tokenizer import create_tokenizer
from ml_playground.error_handling import DataError
from ml_playground.config import DataConfig, PreparerConfig, SharedConfig


"""
Centralized data preparation utilities for ml_playground experiments.

This module provides standardized utilities for data preparation including:
- File state management for tracking changes
- Standardized metadata creation
- Data splitting and encoding utilities
- Atomic file writing operations

All experiments should use these utilities to ensure consistency and proper error handling.
"""


class Encoder(Protocol):
    def encode_ordinary(self, text: str) -> list[int]: ...


class Preparer(Protocol):
    cfg: PreparerConfig

    def __call__(self, shared: SharedConfig) -> None: ...


def snapshot_files(paths: Iterable[Path]) -> dict[Path, tuple[bool, float, int]]:
    """Public utility to take a snapshot of file states for diffing later.

    Returns a dict mapping each path to (exists, mtime, size).
    """
    m: dict[Path, tuple[bool, float, int]] = {}
    for p in paths:
        try:
            if p.exists():
                st = p.stat()
                m[p] = (True, st.st_mtime, st.st_size)
            else:
                m[p] = (False, 0.0, 0)
        except Exception:
            m[p] = (False, 0.0, 0)
    return m


def diff_files(
    paths: Iterable[Path], before: dict[Path, tuple[bool, float, int]]
) -> tuple[set[Path], set[Path], set[Path]]:
    """Public utility to compare file states and determine what changed.

    Returns (created, updated, skipped) as sets of paths.
    """
    after = snapshot_files(paths)
    created, updated, skipped = set(), set(), set()

    all_paths = set(before.keys()) | set(after.keys())

    for p in all_paths:
        b_exists, b_mtime, b_size = before.get(p, (False, 0.0, 0))
        a_exists, a_mtime, a_size = after.get(p, (False, 0.0, 0))

        if not b_exists and a_exists:
            created.add(p)
        elif b_exists and not a_exists:
            # File was deleted, not typically expected in this workflow
            pass
        elif b_exists and a_exists:
            if b_mtime != a_mtime or b_size != a_size:
                updated.add(p)
            else:
                skipped.add(p)

    return created, updated, skipped


def create_standardized_metadata(
    tokenizer: Tokenizer, train_tokens: int, val_tokens: int, extras: dict | None = None
) -> dict:
    """Create standardized metadata for dataset preparation.

    Args:
        tokenizer: The tokenizer used for encoding
        train_tokens: Number of tokens in training set
        val_tokens: Number of tokens in validation set
        extras: Additional metadata to include

    Returns:
        Standardized metadata dictionary
    """
    meta: dict[str, Any] = {
        "meta_version": 1,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }

    # Add tokenizer-specific information (mandatory name)
    meta["tokenizer"] = tokenizer.name

    # Add encoding information if available
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        meta["has_encode"] = True
        meta["has_decode"] = True

    # Include extras if provided
    if extras:
        meta.update(extras)

    return meta


class _PreparerInstance:
    """
    Instance-based Preparer that captures behavior via the provided PreparerConfig.
    Assumes the cfg is already valid and fully resolved by the CLI.
    """

    def __init__(self, cfg: PreparerConfig) -> None:
        self.cfg = cfg

    def __call__(self, shared: SharedConfig) -> None:
        extras = self.cfg.extras

        # Create tokenizer based on config
        tokenizer_type = extras.get("tokenizer_type", "char")
        tokenizer = create_tokenizer(tokenizer_type)

        text = extras.get("raw_text")
        if text is None:
            text = Path(extras["raw_text_path"]).read_text(encoding="utf-8")

        train_arr, val_arr, meta = self._prepare_with_tokenizer(text, tokenizer)
        # Use SharedConfig dataset_dir as the single source of truth
        ds_dir = shared.dataset_dir
        self._write_bin_and_meta(ds_dir, train_arr, val_arr, meta)  # type: ignore[arg-type]

    def _split_train_val(self, text: str, split: float = 0.9) -> tuple[str, str]:
        n = len(text)
        train_end = int(n * split)
        return text[:train_end], text[train_end:]

    def _prepare_with_tokenizer(
        self, text: str, tokenizer: Tokenizer
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        return prepare_with_tokenizer(text, tokenizer)

    def _write_bin_and_meta(
        self, ds_dir: Path, train: np.ndarray, val: np.ndarray, meta: dict
    ) -> None:
        """Write train.bin, val.bin, and meta.pkl atomically and idempotently."""
        # Optionally accept DataConfig via extras to control filenames
        data_cfg = None
        try:
            extras = getattr(self.cfg, "extras", {}) or {}
            maybe_dc = extras.get("data_config")
            if isinstance(maybe_dc, DataConfig):
                data_cfg = maybe_dc
        except Exception:
            data_cfg = None
        write_bin_and_meta(
            ds_dir,
            train,
            val,
            meta,
            logger=getattr(self.cfg, "logger", None),
            data_cfg=data_cfg,
        )


def make_preparer(cfg: PreparerConfig) -> Preparer:
    """
    Factory for an instance-based Preparer. The returned object conforms to the Preparer protocol.
    The CLI constructs this instance and invokes it.
    """
    return _PreparerInstance(cfg)


# Expose config type alongside the protocol for discoverability from the CLI
Preparer.Config = PreparerConfig  # type: ignore[attr-defined]


def split_train_val(text: str, split: float = 0.9) -> tuple[str, str]:
    """Split given text into train/val by ratio (defaults to 90/10).

    Provides a stable functional API used by experiment preparers.
    """
    n = len(text)
    train_end = int(n * split)
    return text[:train_end], text[train_end:]


def create_tokenizer_for_preparation(tokenizer_type: str, **kwargs) -> Tokenizer:
    """Create a tokenizer for data preparation based on type."""
    return create_tokenizer(tokenizer_type, **kwargs)


def prepare_with_tokenizer(
    text: str, tokenizer: Tokenizer, split: float = 0.9
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prepare train/val data and metadata using a tokenizer."""
    # Split text into train/val
    train_text, val_text = split_train_val(text, split)

    # Encode train/val data
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    # Convert to numpy arrays
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    # Create metadata
    meta = create_standardized_metadata(tokenizer, len(train_ids), len(val_ids))

    return train_arr, val_arr, meta


def write_bin_and_meta(
    ds_dir: Path,
    train: np.ndarray,
    val: np.ndarray,
    meta: dict,
    logger: Any | None = None,
    data_cfg: DataConfig | None = None,
) -> None:
    """Write train.bin, val.bin, and meta.pkl atomically and idempotently.

    This is a module-level functional wrapper mirroring _PreparerInstance._write_bin_and_meta
    for experiments that import utilities directly from ml_playground.prepare.
    """
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Use DataConfig-computed paths when provided, otherwise default filenames
    if data_cfg is not None:
        train_path = data_cfg.train_path
        val_path = data_cfg.val_path
        meta_path = data_cfg.meta_path
    else:
        train_path = ds_dir / "train.bin"
        val_path = ds_dir / "val.bin"
        meta_path = ds_dir / "meta.pkl"

    before = snapshot_files([train_path, val_path, meta_path])

    if train_path.exists() and val_path.exists() and meta_path.exists():
        # Strict: meta.pkl must be valid; do not regenerate silently
        try:
            with meta_path.open("rb") as f:
                existing_meta = pickle.load(f)
        except Exception as e:
            raise DataError(
                f"Failed to read existing meta.pkl at {meta_path}: {e}"
            ) from e
        if isinstance(existing_meta, dict) and "meta_version" in existing_meta:
            return
        raise DataError(
            f"Invalid existing meta.pkl at {meta_path}: expected dict with 'meta_version'"
        )

    tmp_train = train_path.with_name("." + train_path.name + ".tmp")
    tmp_val = val_path.with_name("." + val_path.name + ".tmp")
    tmp_meta = meta_path.with_name("." + meta_path.name + ".tmp")

    try:
        tmp_train.write_bytes(train.tobytes())
        tmp_val.write_bytes(val.tobytes())
        with tmp_meta.open("wb") as f:
            pickle.dump(meta, f)

        tmp_train.replace(train_path)
        tmp_val.replace(val_path)
        tmp_meta.replace(meta_path)
    finally:
        # Ensure temporary files are cleaned up on error
        tmp_train.unlink(missing_ok=True)
        tmp_val.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)

    created, updated, skipped = diff_files([train_path, val_path, meta_path], before)

    log_func = None
    if logger and hasattr(logger, "info"):
        log_func = logger.info
    elif logger is None:
        log_func = print

    if log_func:
        try:
            log_func(f"[prepare] Created: {list(created) if created else '[]'}")
            log_func(f"[prepare] Updated: {list(updated) if updated else '[]'}")
            log_func(f"[prepare] Skipped: {list(skipped) if skipped else '[]'}")
        except Exception:
            pass  # Logging should not fail the operation


def seed_text_file(dst: Path, candidates: list[Path]) -> None:
    """Seed a destination text file from the first existing candidate path.

    - Creates parent directories for dst
    - No-ops if dst already exists
    - Raises FileNotFoundError if none of the candidates exist
    """
    if dst.exists():
        return
    for cand in candidates:
        if cand.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(cand.read_bytes())
            return
    raise FileNotFoundError(
        f"seed_text_file: none of the candidate paths exist: {candidates}"
    )


def setup_tokenizer(
    out_dir: Path, data_cfg: DataConfig | None = None
) -> Tokenizer | None:
    """Set up the tokenizer, loading from file if it exists."""
    meta_path = out_dir / "meta.pkl"
    if not meta_path.exists():
        return None
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    tokenizer_type = meta.get("tokenizer_type")
    if tokenizer_type is None:
        raise DataError(f"Invalid meta.pkl at {meta_path}: missing 'tokenizer_type'")
    # Prefer vocab/encoding settings from meta when available
    if tokenizer_type in ("char", "word"):
        vocab = meta.get("stoi") or meta.get("vocab")
        tokenizer = create_tokenizer(tokenizer_type, vocab=vocab)
    elif tokenizer_type == "tiktoken":
        encoding_name = meta.get("encoding_name", "cl100k_base")
        tokenizer = create_tokenizer(tokenizer_type, encoding_name=encoding_name)
    else:
        tokenizer = create_tokenizer(tokenizer_type)
    return tokenizer


# Explicit public API for this module
__all__ = [
    "Encoder",
    "Preparer",
    "snapshot_files",
    "diff_files",
    "create_standardized_metadata",
    "make_preparer",
    "split_train_val",
    "create_tokenizer_for_preparation",
    "prepare_with_tokenizer",
    "write_bin_and_meta",
    "seed_text_file",
    "setup_tokenizer",
]

from __future__ import annotations

from typing import Protocol, Any, Optional, Iterable, Tuple
from pathlib import Path
import pickle
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ml_playground.tokenizer import Tokenizer, create_tokenizer


"""
Centralized data preparation utilities for ml_playground experiments.

This module provides standardized utilities for data preparation including:
- File state management for tracking changes
- Standardized metadata creation
- Data splitting and encoding utilities
- Atomic file writing operations

All experiments should use these utilities to ensure consistency and proper error handling.
"""


class PreparerConfig(BaseModel):
    """Strict config for data preparation (owner-local)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    dataset_dir: Optional[Path] = None
    raw_dir: Optional[Path] = None
    add_structure_tokens: Optional[bool] = None
    doc_separator: Optional[str] = None
    extras: dict[str, Any] = Field(default_factory=dict)
    logger: Any | None = Field(default=None)

    @field_validator("dataset_dir", "raw_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Path | str | None) -> Optional[Path]:
        if v is None:
            return None
        return Path(v)


class Encoder(Protocol):
    def encode_ordinary(self, text: str) -> list[int]: ...


class Preparer(Protocol):
    cfg: PreparerConfig

    def __call__(self) -> None: ...


def _snapshot(paths: Iterable[Path]) -> dict[Path, tuple[bool, float, int]]:
    """Take a snapshot of file states (existence, mtime, size) for diffing later."""
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


def snapshot_files(paths: Iterable[Path]) -> dict[Path, tuple[bool, float, int]]:
    """Public utility to take a snapshot of file states for diffing later.

    Returns a dict mapping each path to (exists, mtime, size).
    """
    return _snapshot(paths)


def _diff(
    paths: Iterable[Path], before: dict[Path, tuple[bool, float, int]]
) -> tuple[list[Path], list[Path], list[Path]]:
    """Compare file states before and after an operation to determine what changed."""
    after = _snapshot(paths)
    created: list[Path] = []
    updated: list[Path] = []
    skipped: list[Path] = []
    for p in paths:
        b = before.get(p, (False, 0.0, 0))
        a = after.get(p, (False, 0.0, 0))
        if not b[0] and a[0]:
            created.append(p)
        elif b[0] and a[0] and a[1] > b[1]:
            updated.append(p)
        elif b[0] and a[0]:
            skipped.append(p)
    return created, updated, skipped


def diff_files(
    paths: Iterable[Path], before: dict[Path, tuple[bool, float, int]]
) -> Tuple[set[Path], set[Path], set[Path]]:
    """Public utility to compare file states and determine what changed.

    Returns (created, updated, skipped) as sets of paths.
    """
    after = _snapshot(paths)
    created, updated, skipped = set(), set(), set()

    for p in paths:
        if not before.get(p, (False, 0.0, 0))[0]:  # didn't exist before
            if after.get(p, (False, 0.0, 0))[0]:  # exists now
                created.add(p)
        else:  # existed before
            if not after.get(p, (False, 0.0, 0))[0]:  # doesn't exist now
                # This case shouldn't happen in normal usage
                pass
            elif (
                before[p][1] != after[p][1]  # mtime changed
                or before[p][2] != after[p][2]  # size changed
            ):
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

    def __call__(self) -> None:
        extras = self.cfg.extras

        # Create tokenizer based on config
        tokenizer_type = extras.get("tokenizer_type", "char")
        tokenizer = create_tokenizer(tokenizer_type)

        text = extras.get("raw_text")
        if text is None:
            text = Path(extras["raw_text_path"]).read_text(encoding="utf-8")

        train_arr, val_arr, meta = self._prepare_with_tokenizer(text, tokenizer)
        self._write_bin_and_meta(self.cfg.dataset_dir, train_arr, val_arr, meta)  # type: ignore[arg-type]

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
        ds_dir.mkdir(parents=True, exist_ok=True)

        train_path = ds_dir / "train.bin"
        val_path = ds_dir / "val.bin"
        meta_path = ds_dir / "meta.pkl"

        before = _snapshot([train_path, val_path, meta_path])

        if train_path.exists() and val_path.exists() and meta_path.exists():
            try:
                with meta_path.open("rb") as f:
                    existing_meta = pickle.load(f)
                if isinstance(existing_meta, dict) and "meta_version" in existing_meta:
                    return
                else:
                    logger = getattr(self.cfg, "logger", None)
                    msg = f"[prepare] Detected invalid meta.pkl at {meta_path}; regenerating dataset artifacts."
                    if logger is not None:
                        try:
                            logger.warning(msg)
                        except Exception:
                            pass
            except Exception:
                logger = getattr(self.cfg, "logger", None)
                msg = f"[prepare] Could not read existing meta.pkl at {meta_path}; regenerating dataset artifacts."
                if logger is not None:
                    try:
                        logger.warning(msg)
                    except Exception:
                        pass
                else:
                    print(msg)

        tmp_train = ds_dir / ".train.bin.tmp"
        tmp_val = ds_dir / ".val.bin.tmp"
        tmp_meta = ds_dir / ".meta.pkl.tmp"

        tmp_train.write_bytes(train.tobytes())
        tmp_val.write_bytes(val.tobytes())
        with tmp_meta.open("wb") as f:
            pickle.dump(meta, f)

        tmp_train.replace(train_path)
        tmp_val.replace(val_path)
        tmp_meta.replace(meta_path)

        created, updated, skipped = _diff([train_path, val_path, meta_path], before)
        logger = getattr(self.cfg, "logger", None)
        if logger is not None:
            try:
                logger.info(f"[prepare] Created: {created}")
                logger.info(f"[prepare] Updated: {updated}")
                logger.info(f"[prepare] Skipped: {skipped}")
            except Exception:
                pass
        else:
            print(f"[prepare] Created: {created}")
            print(f"[prepare] Updated: {updated}")
            print(f"[prepare] Skipped: {skipped}")


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
    ds_dir: Path, train: np.ndarray, val: np.ndarray, meta: dict
) -> None:
    """Write train.bin, val.bin, and meta.pkl atomically and idempotently.

    This is a module-level functional wrapper mirroring _PreparerInstance._write_bin_and_meta
    for experiments that import utilities directly from ml_playground.prepare.
    """
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_path = ds_dir / "train.bin"
    val_path = ds_dir / "val.bin"
    meta_path = ds_dir / "meta.pkl"

    before = _snapshot([train_path, val_path, meta_path])

    if train_path.exists() and val_path.exists() and meta_path.exists():
        try:
            with meta_path.open("rb") as f:
                existing_meta = pickle.load(f)
            if isinstance(existing_meta, dict) and "meta_version" in existing_meta:
                return
            else:
                print(
                    f"[prepare] Detected invalid meta.pkl at {meta_path}; regenerating dataset artifacts."
                )
        except Exception:
            print(
                f"[prepare] Could not read existing meta.pkl at {meta_path}; regenerating dataset artifacts."
            )

    tmp_train = ds_dir / ".train.bin.tmp"
    tmp_val = ds_dir / ".val.bin.tmp"
    tmp_meta = ds_dir / ".meta.pkl.tmp"

    tmp_train.write_bytes(train.tobytes())
    tmp_val.write_bytes(val.tobytes())
    with tmp_meta.open("wb") as f:
        pickle.dump(meta, f)

    tmp_train.replace(train_path)
    tmp_val.replace(val_path)
    tmp_meta.replace(meta_path)

    created, updated, skipped = _diff([train_path, val_path, meta_path], before)
    print(f"[prepare] Created: {created}")
    print(f"[prepare] Updated: {updated}")
    print(f"[prepare] Skipped: {skipped}")


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

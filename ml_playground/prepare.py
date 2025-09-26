from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterable, Literal, cast

import numpy as np
from ml_playground._file_state import diff_file_states, snapshot_file_states
from ml_playground.config import DataConfig, PreparerConfig, SharedConfig
from ml_playground.error_handling import DataError
from ml_playground.logging_protocol import LoggerLike
from ml_playground.tokenizer import CharTokenizer, WordTokenizer, create_tokenizer
from ml_playground.tokenizer_protocol import Tokenizer


"""Core data preparation utilities shared across experiments."""

TokenizerKind = Literal["char", "word", "tiktoken"]


def _coerce_tokenizer_type(value: str) -> TokenizerKind:
    """Validate and cast raw configuration values to ``TokenizerKind``."""
    if value not in {"char", "word", "tiktoken"}:
        raise DataError(
            "Unsupported tokenizer type. Expected one of {'char', 'word', 'tiktoken'}"
        )
    return cast(TokenizerKind, value)


@dataclass(frozen=True)
class PreparationOutcome:
    created_files: tuple[Path, ...]
    updated_files: tuple[Path, ...]
    skipped_files: tuple[Path, ...]
    metadata: dict[str, Any]


def create_standardized_metadata(
    tokenizer: Tokenizer, train_tokens: int, val_tokens: int, extras: dict | None = None
) -> dict:
    """Create standardized metadata for dataset preparation."""

    meta: dict[str, Any] = {
        "meta_version": 1,
        "tokenizer_type": getattr(tokenizer, "name", None) or "unknown",
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "has_encode": hasattr(tokenizer, "encode"),
        "has_decode": hasattr(tokenizer, "decode"),
    }

    # Backward compatibility key kept for older consumers
    meta["tokenizer"] = meta["tokenizer_type"]

    # Include tokenizer details to support reconstruction during sampling
    try:
        if meta["tokenizer_type"] in ("char", "word"):
            vocab = getattr(tokenizer, "stoi", None)
            if isinstance(vocab, dict) and vocab:
                meta["stoi"] = vocab
        elif meta["tokenizer_type"] == "tiktoken":
            encoding_name = getattr(tokenizer, "encoding_name", None)
            if isinstance(encoding_name, str):
                meta["encoding_name"] = encoding_name
    except (AttributeError, TypeError, ValueError):
        # Metadata enrichment is best-effort; never fail preparation
        pass

    if extras:
        meta.update(extras)

    return meta


class _PreparationPipeline:
    def __init__(self, cfg: PreparerConfig, shared: SharedConfig) -> None:
        self._cfg = cfg
        self._shared = shared
        self._logger = cfg.logger

    @property
    def cfg(self) -> PreparerConfig:
        return self._cfg

    @property
    def shared(self) -> SharedConfig:
        return self._shared

    def run(self) -> PreparationOutcome:
        tokenizer_type = self._resolve_tokenizer_type()
        tokenizer = create_tokenizer(tokenizer_type)
        raw_text = self._load_raw_text()
        return self.prepare_from_text(raw_text, tokenizer)

    def prepare_from_text(
        self,
        text: str,
        tokenizer: Tokenizer,
        *,
        split: float | None = None,
        meta_extras: dict[str, Any] | None = None,
    ) -> PreparationOutcome:
        data_cfg = self._resolve_data_config()
        outputs = self._output_paths(data_cfg)
        before = snapshot_file_states(outputs)

        ratio = float(split) if split is not None else self._default_split()
        train_arr, val_arr, meta, tokenizer = prepare_with_tokenizer(
            text,
            tokenizer,
            split=ratio,
        )

        if meta_extras:
            meta.update(meta_extras)

        write_bin_and_meta(
            self._shared.dataset_dir,
            train_arr,
            val_arr,
            meta,
            logger=self._logger,
            data_cfg=data_cfg,
        )

        created, updated, skipped = diff_file_states(outputs, before)
        return PreparationOutcome(
            created_files=tuple(created),
            updated_files=tuple(updated),
            skipped_files=tuple(skipped),
            metadata=meta,
        )

    def _resolve_tokenizer_type(self) -> TokenizerKind:
        raw_value = self._cfg.extras.get("tokenizer_type", "char")
        return _coerce_tokenizer_type(str(raw_value))

    def _resolve_data_config(self) -> DataConfig | None:
        data_cfg = self._cfg.extras.get("data_config")
        if data_cfg is None:
            return None
        if isinstance(data_cfg, DataConfig):
            return data_cfg
        raise DataError(
            "prepare.extras.data_config must be a DataConfig instance when provided"
        )

    def _default_split(self) -> float:
        raw = self._cfg.extras.get("split")
        if raw is None:
            return 0.9
        try:
            ratio = float(raw)
        except (TypeError, ValueError) as exc:
            raise DataError(f"Invalid split ratio in extras: {raw!r}") from exc
        if not (0.0 <= ratio <= 1.0):
            raise DataError(f"split ratio must be within [0.0, 1.0]; received {ratio}")
        return ratio

    def _load_raw_text(self) -> str:
        text = self._cfg.extras.get("raw_text")
        if text is not None:
            return str(text)

        raw_text_path = self._cfg.extras.get("raw_text_path")
        if raw_text_path is not None:
            return Path(raw_text_path).read_text(encoding="utf-8")

        raise DataError("No raw text or path provided in preparer extras")

    def _output_paths(self, data_cfg: DataConfig | None) -> list[Path]:
        if data_cfg is not None:
            return [
                data_cfg.train_path(self._shared.dataset_dir),
                data_cfg.val_path(self._shared.dataset_dir),
                data_cfg.meta_path(self._shared.dataset_dir),
            ]
        return [
            self._shared.dataset_dir / "train.bin",
            self._shared.dataset_dir / "val.bin",
            self._shared.dataset_dir / "meta.pkl",
        ]

    def output_snapshot(self, paths: Iterable[Path]) -> dict:
        return snapshot_file_states(paths)


def create_pipeline(cfg: PreparerConfig, shared: SharedConfig) -> _PreparationPipeline:
    return _PreparationPipeline(cfg, shared)


def split_train_val(text: str, split: float = 0.9) -> tuple[str, str]:
    """Split given text into train/val by ratio (defaults to 90/10).

    Provides a stable functional API used by experiment preparers.
    """
    n = len(text)
    train_end = int(n * split)
    return text[:train_end], text[train_end:]


def prepare_with_tokenizer(
    text: str, tokenizer: Tokenizer, split: float = 0.9
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Tokenizer]:
    """Tokenize raw text into dataset artifacts and metadata.

    Args:
        text: Complete corpus that will be split into train and validation sets.
        tokenizer: Concrete tokenizer implementation used for encoding.
        split: Fraction of tokens assigned to the training split.

    Returns:
        Tuple containing train tokens, validation tokens, metadata dictionary, and the tokenizer used.
    """
    # Split text into train/val
    train_text, val_text = split_train_val(text, split)

    # Build vocab for char/word tokenizers and recreate tokenizer with vocab
    if isinstance(tokenizer, (CharTokenizer, WordTokenizer)):
        all_text = train_text + val_text
        if isinstance(tokenizer, CharTokenizer):
            chars = sorted(set(all_text))
            vocab = {ch: i for i, ch in enumerate(chars)}
            tokenizer = create_tokenizer("char", vocab=vocab)
        elif isinstance(tokenizer, WordTokenizer):
            # For word tokenizer, split into words and build vocab
            import re

            words = re.findall(r"\w+|[^\w\s]", all_text)
            unique_words = sorted(set(words))
            vocab = {word: i for i, word in enumerate(unique_words)}
            tokenizer = create_tokenizer("word", vocab=vocab)

    # Encode train/val data
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    # Convert to numpy arrays
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    # Create metadata
    meta = create_standardized_metadata(tokenizer, len(train_ids), len(val_ids))

    return train_arr, val_arr, meta, tokenizer


def write_bin_and_meta(
    ds_dir: Path,
    train: np.ndarray,
    val: np.ndarray,
    meta: dict,
    logger: LoggerLike,
    data_cfg: DataConfig | None = None,
) -> None:
    """Write train.bin, val.bin, and meta.pkl atomically and idempotently.

    This is a module-level functional wrapper mirroring _PreparerInstance._write_bin_and_meta
    for experiments that import utilities directly from ml_playground.prepare.
    """
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Use DataConfig-computed paths when provided, otherwise default filenames
    if data_cfg is not None:
        train_path = data_cfg.train_path(ds_dir)
        val_path = data_cfg.val_path(ds_dir)
        meta_path = data_cfg.meta_path(ds_dir)
    else:
        train_path = ds_dir / "train.bin"
        val_path = ds_dir / "val.bin"
        meta_path = ds_dir / "meta.pkl"

    before = snapshot_file_states([train_path, val_path, meta_path])

    if train_path.exists() and val_path.exists() and meta_path.exists():
        # Validate existing meta; do not mutate existing artifacts
        try:
            with meta_path.open("rb") as f:
                existing_meta = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            raise DataError(
                f"Failed to read existing meta.pkl at {meta_path}: {e}"
            ) from e
        if isinstance(existing_meta, dict) and "meta_version" in existing_meta:
            # Production policy: do not mutate existing artifacts. If meta is valid, report status and return.
            created, updated, skipped = diff_file_states(
                [train_path, val_path, meta_path], before
            )
            try:
                logger.info(f"[prepare] Created: {list(created) if created else '[]'}")
                logger.info(f"[prepare] Skipped: {list(skipped) if skipped else '[]'}")
            except (OSError, ValueError, TypeError):
                pass  # Logging should not fail the operation
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

    created, updated, skipped = diff_file_states(
        [train_path, val_path, meta_path], before
    )

    try:
        logger.info(f"[prepare] Created: {list(created) if created else '[]'}")
        logger.info(f"[prepare] Skipped: {list(skipped) if skipped else '[]'}")
    except (OSError, ValueError, TypeError):
        pass


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
        tokenizer = create_tokenizer(
            _coerce_tokenizer_type(tokenizer_type), vocab=vocab
        )
    elif tokenizer_type == "tiktoken":
        encoding_name = meta.get("encoding_name", "cl100k_base")
        tokenizer = create_tokenizer(tokenizer_type, encoding_name=encoding_name)
    else:
        tokenizer = create_tokenizer(_coerce_tokenizer_type(tokenizer_type))
    return tokenizer


# Explicit public API for this module
__all__ = [
    "PreparationOutcome",
    "create_pipeline",
    "create_standardized_metadata",
    "split_train_val",
    "prepare_with_tokenizer",
    "write_bin_and_meta",
    "seed_text_file",
    "setup_tokenizer",
    "snapshot_file_states",
    "diff_file_states",
]

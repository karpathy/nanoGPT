"""I/O utilities for persisting dataset artifacts and tokenizer metadata."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from ml_playground.core.file_state import diff_file_states, snapshot_file_states
from ml_playground.configuration.models import DataConfig
from ml_playground.core.error_handling import DataError
from ml_playground.core.logging_protocol import LoggerLike
from ml_playground.data_pipeline.transforms.tokenization import coerce_tokenizer_type
from ml_playground.core.tokenizer import create_tokenizer
from ml_playground.core.tokenizer_protocol import Tokenizer

__all__ = [
    "write_bin_and_meta",
    "seed_text_file",
    "setup_tokenizer",
    "snapshot_file_states",
    "diff_file_states",
]


def write_bin_and_meta(
    ds_dir: Path,
    train: np.ndarray,
    val: np.ndarray,
    meta: dict,
    logger: LoggerLike,
    data_cfg: DataConfig | None = None,
) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)

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
        try:
            with meta_path.open("rb") as f:
                existing_meta = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            raise DataError(
                f"Failed to read existing meta.pkl at {meta_path}: {e}",
                reason=f"Unable to deserialize metadata due to {e.__class__.__name__}",
                rationale="Preparation requires re-using valid metadata to guarantee deterministic outputs",
            ) from e
        if isinstance(existing_meta, dict) and "meta_version" in existing_meta:
            created, _, skipped = diff_file_states(
                [train_path, val_path, meta_path], before
            )
            try:
                logger.info(f"[prepare] Created: {list(created) if created else '[]'}")
                logger.info(f"[prepare] Skipped: {list(skipped) if skipped else '[]'}")
            except (OSError, ValueError, TypeError):
                pass
            return
        raise DataError(
            f"Invalid existing meta.pkl at {meta_path}: expected dict with 'meta_version'",
            reason="Metadata structure missing required 'meta_version' key",
            rationale="Prepare reruns rely on versioned metadata to decide reuse behaviour",
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
        tmp_train.unlink(missing_ok=True)
        tmp_val.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)

    created, _, skipped = diff_file_states([train_path, val_path, meta_path], before)

    try:
        logger.info(f"[prepare] Created: {list(created) if created else '[]'}")
        logger.info(f"[prepare] Skipped: {list(skipped) if skipped else '[]'}")
    except (OSError, ValueError, TypeError):
        pass


def seed_text_file(dst: Path, candidates: list[Path]) -> None:
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
    meta_path = out_dir / "meta.pkl"
    if not meta_path.exists():
        return None
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    tokenizer_type = meta.get("tokenizer_type")
    if tokenizer_type is None:
        raise DataError(
            f"Invalid meta.pkl at {meta_path}: missing 'tokenizer_type'",
            reason="Metadata lacks tokenizer_type entry",
            rationale="Downstream steps need explicit tokenizer kind to construct compatible tokenizers",
        )
    if tokenizer_type in ("char", "word"):
        vocab = meta.get("stoi") or meta.get("vocab")
        tokenizer = create_tokenizer(coerce_tokenizer_type(tokenizer_type), vocab=vocab)
    elif tokenizer_type == "tiktoken":
        encoding_name = meta.get("encoding_name", "cl100k_base")
        tokenizer = create_tokenizer(tokenizer_type, encoding_name=encoding_name)
    else:
        tokenizer = create_tokenizer(coerce_tokenizer_type(tokenizer_type))
    return tokenizer

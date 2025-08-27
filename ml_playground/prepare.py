from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any
from pathlib import Path
import pickle
import numpy as np


class Encoder(Protocol):
    def encode_ordinary(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class PreparerConfig:
    """
    Minimal, frozen config for data preparation. Assumes values are valid/resolved by the CLI.
    """

    dataset_dir: Path | None = None
    raw_dir: Path | None = None
    add_structure_tokens: bool | None = None
    doc_separator: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    logger: Any | None = None


class Preparer(Protocol):
    cfg: PreparerConfig

    def __call__(self) -> None: ...


class _PreparerInstance:
    """
    Instance-based Preparer that captures behavior via the provided PreparerConfig.
    Assumes the cfg is already valid and fully resolved by the CLI.
    """

    def __init__(self, cfg: PreparerConfig) -> None:
        self.cfg = cfg

    def __call__(self) -> None:
        extras = self.cfg.extras
        enc: Encoder = extras["encoder"]  # provided by CLI

        text = extras.get("raw_text")
        if text is None:
            text = Path(extras["raw_text_path"]).read_text(encoding="utf-8")

        train_arr, val_arr = self._prepare_with_encoder(text, enc)
        meta = extras.get("meta") or {"meta_version": 1}
        self._write_bin_and_meta(self.cfg.dataset_dir, train_arr, val_arr, meta)  # type: ignore[arg-type]

    def _split_train_val(self, text: str, split: float = 0.9) -> tuple[str, str]:
        n = len(text)
        return text[: int(n * split)], text[int(n * split) :]

    def _encode_split_with_encoder(
        self, train_data: str, val_data: str, enc: Encoder
    ) -> tuple[np.ndarray, np.ndarray]:
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        train_arr = np.array(train_ids, dtype=np.uint16)
        val_arr = np.array(val_ids, dtype=np.uint16)
        return train_arr, val_arr

    def _prepare_with_encoder(
        self, text: str, enc: Encoder
    ) -> tuple[np.ndarray, np.ndarray]:
        train_data, val_data = self._split_train_val(text, 0.9)
        return self._encode_split_with_encoder(train_data, val_data, enc)

    def _write_bin_and_meta(
        self, ds_dir: Path, train: np.ndarray, val: np.ndarray, meta: dict
    ) -> None:
        """Write train.bin, val.bin, and meta.pkl atomically and idempotently."""
        ds_dir.mkdir(parents=True, exist_ok=True)

        train_path = ds_dir / "train.bin"
        val_path = ds_dir / "val.bin"
        meta_path = ds_dir / "meta.pkl"

        if train_path.exists() and val_path.exists() and meta_path.exists():
            # If existing meta is valid (strict), no-op; otherwise, rewrite artifacts
            try:
                with meta_path.open("rb") as f:
                    existing_meta = pickle.load(f)
                if isinstance(existing_meta, dict) and "meta_version" in existing_meta:
                    return
                else:
                    logger = getattr(self.cfg, "logger", None)
                    msg = (
                        f"[prepare] Detected invalid meta.pkl at {meta_path}; regenerating dataset artifacts."
                    )
                    if logger is not None:
                        try:
                            logger.warning(msg)
                        except Exception:
                            pass
                    else:
                        print(msg)
            except Exception:
                logger = getattr(self.cfg, "logger", None)
                msg = (
                    f"[prepare] Could not read existing meta.pkl at {meta_path}; regenerating dataset artifacts."
                )
                if logger is not None:
                    try:
                        logger.warning(msg)
                    except Exception:
                        pass
                else:
                    print(msg)

        # Write to temp then rename (atomic on POSIX)
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

        print("Wrote:", train_path, val_path, meta_path)


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
    if split <= 0.0:
        split = 0.5
    elif split >= 1.0:
        split = 0.9
    i = int(n * split)
    return text[:i], text[i:]


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

    print("Wrote:", train_path, val_path, meta_path)


def seed_text_file(dst: Path, candidates: list[Path]) -> None:
    """Seed a destination text file from the first existing candidate path.

    - Creates parent directories for dst
    - No-ops if dst already exists
    - Raises FileNotFoundError if none of the candidates exist
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    for p in candidates:
        try:
            if p.exists():
                text = p.read_text(encoding="utf-8")
                dst.write_text(text, encoding="utf-8")
                return
        except Exception:
            # Try next candidate
            continue

    raise FileNotFoundError(
        "No seed text found among candidates: " + ", ".join(str(p) for p in candidates)
    )

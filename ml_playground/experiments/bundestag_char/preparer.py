from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Tuple
import pickle
import numpy as np
from ml_playground.prepare import PreparerConfig, seed_text_file, split_train_val, write_bin_and_meta
from ml_playground.config import load_toml
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)


class BundestagCharPreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        # Track side effects for standard outputs
        exp_dir = Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        outputs = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]

        # Fast-path: if dataset artifacts already exist and meta looks valid, skip heavy work
        if _artifacts_look_valid(outputs):
            msgs = (
                f"[bundestag_char] dataset already prepared at {ds_dir}; skipping.",
                f"[bundestag_char.outputs.created] []",
                f"[bundestag_char.outputs.updated] []",
                f"[bundestag_char.outputs.skipped] {[str(p) for p in outputs]}",
            )
            return PrepareReport(
                created_files=tuple(),
                updated_files=tuple(),
                skipped_files=tuple(outputs),
                messages=msgs,
            )

        pre = _snapshot(outputs)

        # Inline legacy prepare logic
        input_file_path = ds_dir / "input.txt"
        bundled = Path(__file__).parent / "input.txt"
        candidates = [
            ds_dir / "input.txt",
            exp_dir / "page1.txt",
            bundled,
        ]
        seed_text_file(input_file_path, candidates)

        data = input_file_path.read_text(encoding="utf-8")
        # Read n-gram size from experiment config if available
        n: int = 1
        try:
            cfg_path = exp_dir / "config.toml"
            if cfg_path.exists():
                app = load_toml(cfg_path)
                if app.train is not None and app.train.data is not None:
                    n = int(getattr(app.train.data, "ngram_size", 1))
        except Exception:
            n = 1
        if n < 1:
            n = 1

        stoi, itos = _build_vocab(data, n)
        train_text, val_text = split_train_val(data, 0.9)
        train_ids = _encode_ngrams(train_text, stoi, n)
        val_ids = _encode_ngrams(val_text, stoi, n)

        vocab_size = int(len(stoi))
        use_uint16 = vocab_size <= 65535
        dtype_str = "uint16" if use_uint16 else "uint32"
        np_dtype = np.uint16 if use_uint16 else np.uint32

        train_arr = np.array(train_ids, dtype=np_dtype)
        val_arr = np.array(val_ids, dtype=np_dtype)

        meta = {
            "meta_version": 1,
            "kind": "char" if n == 1 else "char_ngram",
            "dtype": dtype_str,
            "stoi": stoi,
            "itos": itos,
            "vocab_size": vocab_size,
        }
        if n != 1:
            meta["ngram_size"] = n

        write_bin_and_meta(ds_dir, train_arr, val_arr, meta)

        created, updated, skipped = _diff(outputs, pre)
        msgs = (
            f"[bundestag_char] prepared dataset at {ds_dir}",
            f"[bundestag_char.outputs.created] {[str(p) for p in created]}",
            f"[bundestag_char.outputs.updated] {[str(p) for p in updated]}",
            f"[bundestag_char.outputs.skipped] {[str(p) for p in skipped]}",
        )
        return PrepareReport(
            created_files=tuple(created),
            updated_files=tuple(updated),
            skipped_files=tuple(skipped),
            messages=msgs,
        )


def _snapshot(paths: Iterable[Path]) -> dict[Path, tuple[bool, float, int]]:
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


def _diff(paths: Iterable[Path], before: dict[Path, tuple[bool, float, int]]):
    created: list[Path] = []
    updated: list[Path] = []
    skipped: list[Path] = []
    for p in paths:
        existed, mtime, size = before.get(p, (False, 0.0, 0))
        try:
            if p.exists():
                st = p.stat()
                if not existed:
                    created.append(p)
                elif st.st_mtime != mtime or st.st_size != size:
                    updated.append(p)
                else:
                    skipped.append(p)
        except Exception:
            # If stat fails post, consider as updated when present
            if p.exists() and not existed:
                created.append(p)
    return created, updated, skipped


def _build_vocab(text: str, n: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    if n <= 1:
        tokens = sorted(set(text))
    else:
        tokens = sorted({text[i : i + n] for i in range(0, max(0, len(text) - n + 1))})
    stoi = {tok: i for i, tok in enumerate(tokens)}
    itos = {i: tok for i, tok in enumerate(tokens)}
    return stoi, itos


def _encode_ngrams(text: str, stoi: Dict[str, int], n: int) -> list[int]:
    if n <= 1:
        return [stoi[c] for c in text if c in stoi]
    L = len(text)
    if L < n:
        return []
    ids: list[int] = []
    for i in range(0, L - n + 1):
        tok = text[i : i + n]
        idx = stoi.get(tok)
        if idx is not None:
            ids.append(idx)
    return ids


def _artifacts_look_valid(outputs: Iterable[Path]) -> bool:
    paths = list(outputs)
    if len(paths) != 3:
        return False
    train_path, val_path, meta_path = paths
    try:
        if not (train_path.exists() and val_path.exists() and meta_path.exists()):
            return False
        with meta_path.open("rb") as f:
            meta_obj = pickle.load(f)
        return isinstance(meta_obj, dict) and ("meta_version" in meta_obj)
    except Exception:
        return False

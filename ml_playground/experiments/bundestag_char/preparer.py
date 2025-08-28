from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Tuple
import pickle
from array import array
from timeit import default_timer as timer
from ml_playground.prepare import PreparerConfig, seed_text_file
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
                "[bundestag_char.outputs.created] []",
                "[bundestag_char.outputs.updated] []",
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
            Path(
                "/Users/tv/code/nanoGPT/ml_playground/experiments/speakger/raw/Bundestag.csv"
            ),
            ds_dir / "input.txt",
            exp_dir / "page1.txt",
            bundled,
        ]
        seed_text_file(input_file_path, candidates)

        # Perform a memory-efficient two-pass preparation:
        # 1) Scan to collect token set and total token count (so we can split train/val).
        # 2) Build vocab (stoi/itos) and stream-encode tokens writing binary files in chunks.
        # This avoids loading the entire file or large Python lists into memory.
        # Read n-gram size from experiment config if available
        n: int = 1
        try:
            cfg_path = exp_dir / "config.toml"
            if cfg_path.exists():
                import tomllib as _tomllib
                with cfg_path.open("rb") as _f:
                    _raw = _tomllib.load(_f)
                if isinstance(_raw, dict):
                    _tr = _raw.get("train")
                    if isinstance(_tr, dict):
                        _dt = _tr.get("data")
                        if isinstance(_dt, dict) and "ngram_size" in _dt:
                            n = int(_dt.get("ngram_size", 1))
        except Exception:
            n = 1
        if n < 1:
            n = 1

        # First pass: gather token set and total token count with progress reporting.
        token_set: set[str] = set()
        total_tokens = 0
        chunk_size = 64 * 1024  # 64KB reads

        total_bytes = None
        try:
            total_bytes = input_file_path.stat().st_size
        except Exception:
            total_bytes = None

        print(f"[prepare] First pass: scanning {input_file_path} for tokens...")
        start_time = timer()
        last_report = start_time
        report_interval_seconds = 5.0
        processed_bytes = 0

        if n <= 1:
            # Use binary reads for accurate byte progress, decode for token extraction
            with input_file_path.open("rb") as f:
                while True:
                    chunk_bytes = f.read(chunk_size)
                    if not chunk_bytes:
                        break
                    processed_bytes = f.tell()
                    chunk = chunk_bytes.decode("utf-8", errors="ignore")
                    token_set.update(chunk)
                    total_tokens += len(chunk)
                    now = timer()
                    if now - last_report >= report_interval_seconds:
                        last_report = now
                        if total_bytes:
                            pct = processed_bytes * 100.0 / total_bytes
                            print(
                                f"[prepare] scan progress: {processed_bytes}/{total_bytes} bytes ({pct:.1f}%), tokens_seen={len(token_set)}"
                            )
                        else:
                            print(
                                f"[prepare] scan progress: bytes_processed={processed_bytes}, tokens_seen={len(token_set)}"
                            )
        else:
            # For n-grams keep a sliding tail across binary chunk boundaries
            with input_file_path.open("rb") as f:
                tail = ""
                while True:
                    chunk_bytes = f.read(chunk_size)
                    if not chunk_bytes:
                        break
                    processed_bytes = f.tell()
                    chunk = chunk_bytes.decode("utf-8", errors="ignore")
                    seq = tail + chunk
                    L = len(seq)
                    if L >= n:
                        for i in range(0, L - n + 1):
                            token_set.add(seq[i : i + n])
                        total_tokens += max(0, L - n + 1)
                        tail = seq[-(n - 1) :]
                    else:
                        tail = seq
                    now = timer()
                    if now - last_report >= report_interval_seconds:
                        last_report = now
                        if total_bytes:
                            pct = processed_bytes * 100.0 / total_bytes
                            print(
                                f"[prepare] scan progress: {processed_bytes}/{total_bytes} bytes ({pct:.1f}%), ngram_types={len(token_set)}"
                            )
                        else:
                            print(
                                f"[prepare] scan progress: bytes_processed={processed_bytes}, ngram_types={len(token_set)}"
                            )

        elapsed = timer() - start_time
        print(
            f"[prepare] First pass complete: discovered {len(token_set)} unique tokens, total_tokens_estimate={total_tokens}, time={elapsed:.1f}s"
        )

        # Build vocab mappings
        tokens_sorted = sorted(token_set)
        stoi = {tok: i for i, tok in enumerate(tokens_sorted)}
        itos = {i: tok for i, tok in enumerate(tokens_sorted)}
        vocab_size = int(len(stoi))

        use_uint16 = vocab_size <= 65535
        dtype_str = "uint16" if use_uint16 else "uint32"
        typecode = "H" if use_uint16 else "I"  # for array('H') or array('I')

        # Compute split point in token counts
        train_target = int(total_tokens * 0.9)

        # Prepare temp file paths
        tmp_train = ds_dir / ".train.bin.tmp"
        tmp_val = ds_dir / ".val.bin.tmp"
        tmp_meta = ds_dir / ".meta.pkl.tmp"

        # Ensure dataset dir exists
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Second pass: stream-encode tokens and write binary in chunks with progress reporting
        buf_size = 32_768  # flush buffer after this many tokens
        train_written = 0
        val_written = 0
        train_buf = array(typecode)
        val_buf = array(typecode)

        def _flush_buf(ar: array, fh, name: str = "?"):
            if len(ar) == 0:
                return
            ar.tofile(fh)
            ar_len = len(ar)
            ar.clear()
            print(f"[prepare] flushed {ar_len} items to {name}")

        print(
            f"[prepare] Second pass: encoding and writing to {tmp_train} and {tmp_val}..."
        )
        start_time = timer()
        last_report = start_time
        report_interval_seconds = 5.0
        processed_tokens = 0

        with tmp_train.open("wb") as train_fh, tmp_val.open("wb") as val_fh:
            if n <= 1:
                with input_file_path.open("rb") as f:
                    while True:
                        chunk_bytes = f.read(chunk_size)
                        if not chunk_bytes:
                            break
                        chunk = chunk_bytes.decode("utf-8", errors="ignore")
                        for ch in chunk:
                            idx = stoi.get(ch)
                            if idx is None:
                                continue
                            processed_tokens += 1
                            if train_written < train_target:
                                train_buf.append(idx)
                                train_written += 1
                                if len(train_buf) >= buf_size:
                                    _flush_buf(train_buf, train_fh, "train")
                            else:
                                val_buf.append(idx)
                                val_written += 1
                                if len(val_buf) >= buf_size:
                                    _flush_buf(val_buf, val_fh, "val")
                        now = timer()
                        if now - last_report >= report_interval_seconds:
                            last_report = now
                            pct = (
                                (processed_tokens / total_tokens * 100.0)
                                if total_tokens
                                else 0.0
                            )
                            print(
                                f"[prepare] encode progress: tokens_processed={processed_tokens}, train={train_written}, val={val_written}, {pct:.1f}%"
                            )
            else:
                with input_file_path.open("rb") as f:
                    tail = ""
                    while True:
                        chunk_bytes = f.read(chunk_size)
                        if not chunk_bytes:
                            break
                        chunk = chunk_bytes.decode("utf-8", errors="ignore")
                        seq = tail + chunk
                        L = len(seq)
                        if L >= n:
                            for i in range(0, L - n + 1):
                                tok = seq[i : i + n]
                                idx = stoi.get(tok)
                                if idx is None:
                                    continue
                                processed_tokens += 1
                                if train_written < train_target:
                                    train_buf.append(idx)
                                    train_written += 1
                                    if len(train_buf) >= buf_size:
                                        _flush_buf(train_buf, train_fh, "train")
                                else:
                                    val_buf.append(idx)
                                    val_written += 1
                                    if len(val_buf) >= buf_size:
                                        _flush_buf(val_buf, val_fh, "val")
                            tail = seq[-(n - 1) :]
                        else:
                            tail = seq
                        now = timer()
                        if now - last_report >= report_interval_seconds:
                            last_report = now
                            pct = (
                                (processed_tokens / total_tokens * 100.0)
                                if total_tokens
                                else 0.0
                            )
                            print(
                                f"[prepare] encode progress: tokens_processed={processed_tokens}, train={train_written}, val={val_written}, {pct:.1f}%"
                            )
            # flush remaining buffers
            _flush_buf(train_buf, train_fh, "train")
            _flush_buf(val_buf, val_fh, "val")

        elapsed = timer() - start_time
        print(
            f"[prepare] Second pass complete: train_written={train_written}, val_written={val_written}, time={elapsed:.1f}s"
        )

        # Atomically write meta and rename temp bins to final names
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

        # write meta to temp and replace atomically
        with tmp_meta.open("wb") as fw:
            pickle.dump(meta, fw)

        (tmp_train).replace(ds_dir / "train.bin")
        (tmp_val).replace(ds_dir / "val.bin")
        (tmp_meta).replace(ds_dir / "meta.pkl")

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

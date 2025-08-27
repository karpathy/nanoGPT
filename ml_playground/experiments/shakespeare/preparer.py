from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import requests
import tiktoken
from ml_playground.prepare import PreparerConfig, split_train_val, write_bin_and_meta
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class ShakespearePreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        exp_dir = Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        outputs = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]
        pre = _snapshot(outputs)

        # Inline legacy logic: download input if needed, tokenize via tiktoken, write arrays+meta
        f_input = ds_dir / "input.txt"
        f_train = ds_dir / "train.bin"
        f_val = ds_dir / "val.bin"

        if not f_input.exists():
            try:
                resp = requests.get(DATA_URL, timeout=60)
                resp.raise_for_status()
                f_input.write_text(resp.text, encoding="utf-8")
            except Exception as e:
                raise SystemExit(f"Failed to download Tiny Shakespeare: {e}")

        # If outputs exist and newer than input, skip heavy work
        try:
            if f_train.exists() and f_val.exists():
                t_in = f_input.stat().st_mtime
                if f_train.stat().st_mtime >= t_in and f_val.stat().st_mtime >= t_in:
                    created, updated, skipped = _diff(outputs, pre)
                    msgs = (
                        f"[shakespeare] dataset already prepared at {ds_dir}; skipping.",
                        f"[shakespeare.outputs.created] {[str(p) for p in created]}",
                        f"[shakespeare.outputs.updated] {[str(p) for p in updated]}",
                        f"[shakespeare.outputs.skipped] {[str(p) for p in skipped]}",
                    )
                    return PrepareReport(
                        created_files=tuple(created),
                        updated_files=tuple(updated),
                        skipped_files=tuple(skipped),
                        messages=msgs,
                    )
        except OSError:
            pass

        data = f_input.read_text(encoding="utf-8")
        enc_name = "gpt2"
        enc = tiktoken.get_encoding(enc_name)

        train_text, val_text = split_train_val(data, 0.9)
        train_ids = enc.encode(train_text, allowed_special={"<|endoftext|>"})
        val_ids = enc.encode(val_text, allowed_special={"<|endoftext|>"})

        train_arr = np.array(train_ids, dtype=np.uint32)
        val_arr = np.array(val_ids, dtype=np.uint32)

        meta = {
            "meta_version": 1,
            "kind": "tiktoken",
            "dtype": "uint32",
            "encoding": enc_name,
        }

        write_bin_and_meta(ds_dir, train_arr, val_arr, meta)

        created, updated, skipped = _diff(outputs, pre)
        msgs = (
            f"[shakespeare] prepared dataset at {ds_dir}",
            f"[shakespeare.outputs.created] {[str(p) for p in created]}",
            f"[shakespeare.outputs.updated] {[str(p) for p in updated]}",
            f"[shakespeare.outputs.skipped] {[str(p) for p in skipped]}",
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
            if p.exists() and not existed:
                created.append(p)
    return created, updated, skipped

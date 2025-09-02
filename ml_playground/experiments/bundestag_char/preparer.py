from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Tuple
import pickle
from array import array
from timeit import default_timer as timer
from ml_playground.prepare import PreparerConfig, seed_text_file, split_train_val, prepare_with_tokenizer, write_bin_and_meta, snapshot_files, diff_files, create_standardized_metadata
from ml_playground.tokenizer import create_tokenizer, CharTokenizer
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)
from ml_playground.error_handling import DataError, safe_file_operation, validate_file_exists, ProgressReporter
from ml_playground.config import validate_config_field, validate_path_exists


class BundestagCharPreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        exp_dir = Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        outputs = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]

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

        pre = snapshot_files(outputs)

        input_file_path = ds_dir / "input.txt"
        bundled = Path(__file__).parent / "input.txt"
        candidates = [
            Path("/datasets/Bundestag.csv"),
            ds_dir / "input.txt",
            exp_dir / "page1.txt",
            bundled,
        ]
        seed_text_file(input_file_path, candidates)

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

        validate_config_field(n, "ngram_size", int, min_value=1)
        
        validate_file_exists(input_file_path, "Input text file")

        data = input_file_path.read_text(encoding="utf-8")
        
        if n == 1:
            tokens = sorted(set(data))
            stoi = {tok: i for i, tok in enumerate(tokens)}
            tokenizer = CharTokenizer(vocab=stoi)
        else:
            return self._legacy_prepare(cfg, data, n, ds_dir, pre, outputs)

        train_arr, val_arr, meta = prepare_with_tokenizer(data, tokenizer)
        
        write_bin_and_meta(ds_dir, train_arr, val_arr, meta)
        
        created, updated, skipped = diff_files(outputs, pre)
        
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

    def _legacy_prepare(self, cfg: PreparerConfig, data: str, n: int, ds_dir: Path, pre: dict, outputs: list) -> PrepareReport:
        logger = cfg.logger or logging.getLogger(__name__)
        progress = ProgressReporter(logger, total_steps=3)
        
        progress.start("Starting n-gram tokenization preparation")
        
        vocab = _build_vocab(data, n)
        stoi = {tok: i for i, tok in enumerate(vocab)}
        
        progress.update(1, "Collecting vocabulary")
        
        progress.update(1, "Encoding tokens and writing files")
        train_path, val_path = ds_dir / "train.bin", ds_dir / "val.bin"
        
        train_text, val_text = split_train_val(data)
        
        train_tokens = _encode_ngrams(train_text, stoi, n)
        val_tokens = _encode_ngrams(val_text, stoi, n)
        
        train_arr = array("I", train_tokens)
        val_arr = array("I", val_tokens)
        
        safe_file_operation(lambda: train_arr.tofile(open(train_path, "wb")), logger=logger)
        safe_file_operation(lambda: val_arr.tofile(open(val_path, "wb")), logger=logger)
        
        meta = create_standardized_metadata(
            tokenizer=CharTokenizer(vocab=stoi),
            train_tokens=len(train_tokens),
            val_tokens=len(val_tokens),
            extras={"ngram_size": n}
        )
        
        with ds_dir / "meta.pkl".open("wb") as f:
            pickle.dump(meta, f)
        
        progress.finish("N-gram tokenization preparation completed")
        
        created, updated, skipped = diff_files(outputs, pre)
        
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

    @staticmethod
    def _flush_buf(ar: array, fh, name: str = "?") -> None:
        try:
            ar.tofile(fh)
        except Exception as e:
            raise DataError(f"Failed to write buffer '{name}' to file: {e}") from e
        ar.clear()


def _build_vocab(text: str, n: int) -> list[str]:
    if n == 1:
        return sorted(set(text))
    return sorted(set(text[i : i + n] for i in range(len(text) - n + 1)))


def _encode_ngrams(text: str, stoi: Dict[str, int], n: int) -> list[int]:
    return [stoi[text[i : i + n]] for i in range(len(text) - n + 1)]


def _artifacts_look_valid(outputs: Iterable[Path]) -> bool:
    for path in outputs:
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            return False
    return True

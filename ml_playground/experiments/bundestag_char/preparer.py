from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ml_playground.prepare import (
    PreparerConfig,
    seed_text_file,
    prepare_with_tokenizer,
    snapshot_files,
    diff_files,
    write_bin_and_meta,
)
from ml_playground.tokenizer import CharTokenizer
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)
from ml_playground.error_handling import DataError, validate_file_exists


def ensure_modern_ngram(extras: Mapping[str, Any]) -> None:
    """Validate that no legacy n-gram overrides are provided via preparer extras."""

    raw_n = extras.get("ngram_size")
    if raw_n is None:
        return
    try:
        legacy_value = int(raw_n)
    except (TypeError, ValueError) as exc:
        raise DataError(
            "Legacy n-gram preparation has been removed. Remove the 'ngram_size' extra "
            "and configure tokenizers via experiment config."
        ) from exc
    if legacy_value != 1:
        raise DataError(
            "Legacy n-gram preparation has been removed. Configure tokenizer choices "
            "via experiment config (for example, train.data.tokenizer) and keep "
            "'ngram_size' fixed at 1."
        )


class BundestagCharPreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        exp_dir = Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        outputs = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]

        ensure_modern_ngram(cfg.extras)

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

        validate_file_exists(input_file_path, "Input text file")

        data = input_file_path.read_text(encoding="utf-8")

        tokenizer = CharTokenizer()  # Let prepare_with_tokenizer build the vocab

        train_arr, val_arr, meta, tokenizer = prepare_with_tokenizer(data, tokenizer)

        write_bin_and_meta(ds_dir, train_arr, val_arr, meta, logger=cfg.logger)

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


def _artifacts_look_valid(outputs: Iterable[Path]) -> bool:
    for path in outputs:
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            return False
    return True

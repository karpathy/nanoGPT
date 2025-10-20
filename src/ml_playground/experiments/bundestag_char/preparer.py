from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ml_playground.configuration.models import PreparerConfig
from ml_playground.data_pipeline.transforms.tokenization import prepare_with_tokenizer
from ml_playground.data_pipeline.transforms.io import (
    diff_file_states,
    seed_text_file,
    snapshot_file_states,
    write_bin_and_meta,
)
from ml_playground.core.tokenizer import CharTokenizer
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)
from ml_playground.core.error_handling import validate_file_exists


class BundestagCharPreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        extras = getattr(cfg, "extras", {}) or {}
        base_dir_override = extras.get("dataset_dir_override")
        if base_dir_override is not None:
            exp_dir = Path(base_dir_override)
        else:
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

        pre = snapshot_file_states(outputs)

        input_file_path = ds_dir / "input.txt"
        bundled = Path(__file__).parent / "input.txt"
        candidates = [
            Path("/datasets/Bundestag.csv"),
            ds_dir / "input.txt",
            exp_dir / "input.txt",
            exp_dir / "page1.txt",
            bundled,
        ]
        seed_text_file(input_file_path, candidates)

        validate_file_exists(input_file_path, "Input text file")

        raw_path = cfg.raw_text_path or input_file_path
        data = Path(raw_path).read_text(encoding="utf-8")

        tokenizer_type = (
            cfg.tokenizer_type if hasattr(cfg, "tokenizer_type") else "char"
        )
        if tokenizer_type != "char":
            raise ValueError(
                "BundestagCharPreparer only supports char tokenizer configured via prepare.tokenizer_type"
            )
        tokenizer = CharTokenizer()  # Let prepare_with_tokenizer build the vocab

        train_arr, val_arr, meta, tokenizer = prepare_with_tokenizer(data, tokenizer)

        write_bin_and_meta(ds_dir, train_arr, val_arr, meta, logger=cfg.logger)

        created, updated, skipped = diff_file_states(outputs, pre)

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

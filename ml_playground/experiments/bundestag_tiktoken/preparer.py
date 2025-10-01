from __future__ import annotations

from pathlib import Path
import numpy as np
from ml_playground.configuration.models import PreparerConfig
from ml_playground.data_pipeline.transforms.tokenization import (
    create_standardized_metadata,
    split_train_val,
)
from ml_playground.data_pipeline.transforms.io import (
    diff_file_states,
    seed_text_file,
    snapshot_file_states,
    write_bin_and_meta,
)
from ml_playground.tokenizer import TiktokenTokenizer
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)
from ml_playground.core.error_handling import validate_file_exists, ProgressReporter


class BundestagTiktokenPreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        exp_dir = Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        outputs = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]

        pre = snapshot_file_states(outputs)

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
        train_text, val_text = split_train_val(data)

        logger = cfg.logger
        progress = ProgressReporter(logger, total_steps=4)

        progress.start("Starting Bundestag tiktoken preparation")

        tokenizer = TiktokenTokenizer(encoding_name="gpt2")

        progress.update(1, "Encoding training data")
        train_ids = tokenizer.encode(train_text)
        progress.update(1, "Encoding validation data")
        val_ids = tokenizer.encode(val_text)

        train_ids_arr: np.ndarray = np.array(train_ids, dtype=np.uint16)
        val_ids_arr: np.ndarray = np.array(val_ids, dtype=np.uint16)

        progress.update(1, "Creating metadata")
        meta = create_standardized_metadata(
            tokenizer=tokenizer, train_tokens=len(train_ids), val_tokens=len(val_ids)
        )

        write_bin_and_meta(ds_dir, train_ids_arr, val_ids_arr, meta, logger=cfg.logger)

        progress.finish("Bundestag tiktoken preparation completed")

        created, updated, skipped = diff_file_states(outputs, pre)

        msgs = (
            f"[bundestag_tiktoken] prepared dataset at {ds_dir}",
            f"[bundestag_tiktoken.outputs.created] {[str(p) for p in created]}",
            f"[bundestag_tiktoken.outputs.updated] {[str(p) for p in updated]}",
            f"[bundestag_tiktoken.outputs.skipped] {[str(p) for p in skipped]}",
        )
        return PrepareReport(
            created_files=tuple(created),
            updated_files=tuple(updated),
            skipped_files=tuple(skipped),
            messages=msgs,
        )

from __future__ import annotations

from pathlib import Path
import numpy as np
import requests
import requests.exceptions
from ml_playground.prepare import (
    PreparerConfig,
    split_train_val,
    write_bin_and_meta,
    create_standardized_metadata,
    snapshot_file_states,
    diff_file_states,
)
from ml_playground.tokenizer import create_tokenizer
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)
from ml_playground.error_handling import (
    DataError,
    validate_file_exists,
    ProgressReporter,
)

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class ShakespearePreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        exp_dir = Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        outputs = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]

        pre = snapshot_file_states(outputs)

        f_input = ds_dir / "input.txt"

        if not f_input.exists():
            try:
                resp = requests.get(DATA_URL, timeout=30)
                resp.raise_for_status()
                f_input.write_text(resp.text, encoding="utf-8")
            except requests.exceptions.RequestException as e:
                raise DataError(f"Failed to download Shakespeare dataset: {e}") from e

        validate_file_exists(f_input, "Shakespeare input file")

        data = f_input.read_text(encoding="utf-8")
        train_text, val_text = split_train_val(data)

        logger = cfg.logger
        progress = ProgressReporter(logger, total_steps=4)

        progress.start("Starting Shakespeare dataset preparation")

        tokenizer = create_tokenizer("tiktoken", encoding_name="gpt2")

        progress.update(1, "Creating tokenizer")

        progress.update(1, "Encoding training data")
        train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
        progress.update(1, "Encoding validation data")
        val_ids = tokenizer.encode(val_text)

        train_ids_arr: np.ndarray = np.array(train_ids, dtype=np.uint16)
        val_ids_arr: np.ndarray = np.array(val_ids, dtype=np.uint16)

        progress.update(1, "Creating metadata")
        meta = create_standardized_metadata(tokenizer, len(train_ids), len(val_ids))

        write_bin_and_meta(ds_dir, train_ids_arr, val_ids_arr, meta, logger=cfg.logger)

        progress.finish("Shakespeare dataset preparation completed")

        created, updated, skipped = diff_file_states(outputs, pre)

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

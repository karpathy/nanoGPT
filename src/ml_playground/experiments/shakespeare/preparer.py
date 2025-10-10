from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any, cast
import numpy as np
import requests
import requests.exceptions
from ml_playground.configuration.models import PreparerConfig
from ml_playground.data_pipeline.transforms.tokenization import (
    create_standardized_metadata,
    split_train_val,
)
from ml_playground.data_pipeline.transforms.io import (
    diff_file_states,
    snapshot_file_states,
    write_bin_and_meta,
)
from ml_playground.core.tokenizer import create_tokenizer
from ml_playground.core.tokenizer_protocol import Tokenizer
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)
from ml_playground.core.error_handling import (
    DataError,
    validate_file_exists,
    ProgressReporter,
)

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class ShakespearePreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        # Allow tests to inject a base_dir to avoid patching module __file__
        base_dir = cfg.extras.get("base_dir") if cfg and cfg.extras else None
        exp_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        outputs = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]

        pre = snapshot_file_states(outputs)

        f_input = ds_dir / "input.txt"

        if not f_input.exists():
            # Allow injectable http_get for tests
            http_get = None
            if cfg and cfg.extras:
                http_get = cfg.extras.get("http_get")
            try:
                _get = http_get if callable(http_get) else requests.get
                resp = _get(DATA_URL, timeout=30)
                # Support simple fake objects without raise_for_status
                rfs = getattr(resp, "raise_for_status", None)
                if callable(rfs):
                    rfs()
                text = getattr(resp, "text", None)
                if text is None:
                    raise DataError("http_get did not return an object with .text")
                f_input.write_text(text, encoding="utf-8")
            except requests.exceptions.RequestException as e:
                raise DataError(f"Failed to download Shakespeare dataset: {e}") from e

        validate_file_exists(f_input, "Shakespeare input file")

        data = f_input.read_text(encoding="utf-8")
        train_text, val_text = split_train_val(data)

        # Access logger from cfg
        logger = cfg.logger
        progress = ProgressReporter(logger, total_steps=4)

        progress.start("Starting Shakespeare dataset preparation")

        # Allow injectable tokenizer factory for tests
        tok_factory: Optional[Callable[[], Any]] = None
        if cfg and cfg.extras:
            tf = cfg.extras.get("tokenizer_factory")
            if callable(tf):
                tok_factory = cast(Callable[[], Any], tf)
        tokenizer_obj: Any = (
            tok_factory()
            if tok_factory is not None
            else create_tokenizer("tiktoken", encoding_name="gpt2")
        )

        # Cast to Tokenizer protocol for type-checkers and use it
        tokenizer: Tokenizer = cast(Tokenizer, tokenizer_obj)

        progress.update(1, "Creating tokenizer")

        progress.update(1, "Encoding training data")
        train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
        progress.update(1, "Encoding validation data")
        val_ids = tokenizer.encode(val_text)

        train_ids_arr: np.ndarray = np.array(train_ids, dtype=np.uint16)
        val_ids_arr: np.ndarray = np.array(val_ids, dtype=np.uint16)
        progress.update(1, "Creating metadata")
        meta = create_standardized_metadata(tokenizer, len(train_ids), len(val_ids))

        # Allow injectable writer function for tests
        writer_fn = None
        if cfg and cfg.extras:
            writer_fn = cfg.extras.get("writer_fn")
        if callable(writer_fn):
            writer_fn(ds_dir, train_ids_arr, val_ids_arr, meta, logger=cfg.logger)
        else:
            write_bin_and_meta(
                ds_dir, train_ids_arr, val_ids_arr, meta, logger=cfg.logger
            )

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

from __future__ import annotations


class RemovedImportError(ImportError):
    """Raised when importing a retired compatibility shim."""


def _raise() -> None:  # pragma: no cover - executed on import
    raise RemovedImportError(
        "`ml_playground.prepare` has been retired. Use the canonical modules under "
        "`ml_playground.data_pipeline.preparer` for `PreparationOutcome` and "
        "`create_pipeline`, and `ml_playground.data_pipeline` for helpers such as "
        "`split_train_val`, `prepare_with_tokenizer`, and `write_bin_and_meta`."
    )


_raise()


__all__: list[str] = []

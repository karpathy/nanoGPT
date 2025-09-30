from __future__ import annotations


class RemovedImportError(ImportError):
    """Raised when importing a retired compatibility shim."""


def _raise() -> None:  # pragma: no cover - executed on import
    raise RemovedImportError(
        "`ml_playground.data` has been retired. Use the canonical modules under "
        "`ml_playground.data_pipeline` instead, e.g. "
        "`ml_playground.data_pipeline.sampling` for `SimpleBatches` and "
        "`sample_batch`, or `ml_playground.data_pipeline.sources` for `MemmapReader`."
    )


_raise()


__all__: list[str] = []

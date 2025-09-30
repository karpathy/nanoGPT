from __future__ import annotations


class RemovedImportError(ImportError):
    """Raised when importing a retired compatibility shim."""


def _raise() -> None:  # pragma: no cover - executed on import
    raise RemovedImportError(
        "`ml_playground.sampler` has been retired. Use the canonical module "
        "`ml_playground.sampling` instead for `Sampler` and `sample`."
    )


_raise()


__all__: list[str] = []

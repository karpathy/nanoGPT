from __future__ import annotations

from pathlib import Path
from ml_playground.configuration.models import PreparerConfig
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.toml"


class SpeakGerPreparer(_PreparerProto):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        # Minimal preparer: this experiment expects pre-tokenized data or external pipeline.
        # We simply ensure the dataset directory exists and report no-op if present.
        extras = getattr(cfg, "extras", {}) or {}
        base_dir_override = extras.get("dataset_dir_override")
        exp_dir = (
            Path(base_dir_override)
            if base_dir_override is not None
            else Path(__file__).resolve().parent
        )
        ds_dir = exp_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        msgs = (
            f"[speakger] no-op prepare; expecting external/pre-tokenized dataset at {ds_dir}",
        )
        return PrepareReport(
            created_files=tuple(),
            updated_files=tuple(),
            skipped_files=(ds_dir,),
            messages=msgs,
        )

from __future__ import annotations

from pathlib import Path
from ml_playground.config import TrainerConfig
from ml_playground.experiments.protocol import (
    Trainer as _TrainerProto,
    TrainReport,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.toml"


class SpeakGerTrainer(_TrainerProto):
    def train(self, cfg: TrainerConfig) -> TrainReport:  # type: ignore[override]
        # Minimal trainer placeholder: external fine-tuning pipeline not embedded.
        # Create out_dir if specified to align with downstream expectations.
        out_dir = getattr(getattr(cfg, "runtime", object()), "out_dir", None)
        if isinstance(out_dir, Path):
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return TrainReport(
            created_files=(),
            updated_files=(),
            skipped_files=(),
            messages=(
                "[speakger] no-op trainer; external HF/PEFT pipeline expected outside repository",
            ),
        )

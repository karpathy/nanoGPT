from __future__ import annotations

from pathlib import Path
from ml_playground.config import TrainerConfig
from ml_playground.experiments.protocol import (
    Trainer as _TrainerProto,
    TrainReport,
)


class BundestagQwen15bLoraMpsTrainer(_TrainerProto):
    def train(self, cfg: TrainerConfig) -> TrainReport:  # type: ignore[override]
        """Minimal no-op trainer placeholder for the preset.

        This repository does not embed the full HF/PEFT training loop for Qwen.
        We provide a stub that ensures the configured out_dir exists so that the
        CLI flow can run end-to-end and tests observing side-effects can pass.
        """
        out_dir = getattr(getattr(cfg, "runtime", object()), "out_dir", None)
        if isinstance(out_dir, Path):
            out_dir.mkdir(parents=False, exist_ok=True)
        return TrainReport(
            created_files=(),
            updated_files=(),
            skipped_files=(),
            messages=(
                "[bundestag_qwen15b_lora_mps] no-op trainer; external HF/PEFT pipeline expected outside repository",
            ),
        )

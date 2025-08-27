from __future__ import annotations

from ml_playground.config import TrainerConfig
from ml_playground.experiments.protocol import (
    Trainer as _TrainerProto,
    TrainReport,
)
from ml_playground.trainer import train as _core_train


class BundestagTiktokenTrainer(_TrainerProto):
    def train(self, cfg: TrainerConfig) -> TrainReport:  # type: ignore[override]
        _core_train(cfg)
        return TrainReport(
            created_files=(),
            updated_files=(),
            skipped_files=(),
            messages=("[bundestag_tiktoken] training finished",),
        )

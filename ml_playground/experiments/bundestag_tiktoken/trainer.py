from __future__ import annotations

from pathlib import Path
from ml_playground.config import TrainerConfig, SharedConfig
from ml_playground.experiments.protocol import (
    Trainer as _TrainerProto,
    TrainReport,
)
from ml_playground.training import Trainer as _CoreTrainer


class BundestagTiktokenTrainer(_TrainerProto):
    def train(self, cfg: TrainerConfig) -> TrainReport:  # type: ignore[override]
        out_dir: Path = cfg.runtime.out_dir
        shared = SharedConfig(
            experiment="bundestag_tiktoken",
            config_path=out_dir / "cfg.toml",
            project_home=out_dir.parent,
            dataset_dir=out_dir,
            train_out_dir=out_dir,
            sample_out_dir=out_dir,
        )
        _CoreTrainer(cfg, shared).run()
        return TrainReport(
            created_files=(),
            updated_files=(),
            skipped_files=(),
            messages=("[bundestag_tiktoken] training finished",),
        )

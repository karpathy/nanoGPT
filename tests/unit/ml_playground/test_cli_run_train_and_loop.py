from __future__ import annotations

from pathlib import Path

import pytest

import ml_playground.cli as cli
from ml_playground.prepare import PreparerConfig
from ml_playground.config import (
    TrainerConfig,
    SamplerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
    SampleConfig,
)


def test_run_loop_calls_in_order_and_handles_print_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[str] = []

    # Monkeypatch public module entrypoints called by CLI runners
    import ml_playground.prepare as prepare_mod
    import ml_playground.trainer as trainer_mod
    import ml_playground.sampler as sampler_mod

    def fake_preparer():
        calls.append("prepare")

    monkeypatch.setattr(prepare_mod, "make_preparer", lambda cfg: fake_preparer)
    monkeypatch.setattr(trainer_mod, "train", lambda cfg: calls.append("train"))
    monkeypatch.setattr(sampler_mod, "sample", lambda cfg: calls.append("sample"))

    # Configs
    prep_cfg = PreparerConfig()
    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    scfg = SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=SampleConfig())

    # Should not raise; ensure order prepare -> train -> sample
    cli._run_loop("exp", tmp_path / "cfg.toml", prep_cfg, tcfg, scfg)

    assert calls == ["prepare", "train", "sample"]

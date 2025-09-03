from __future__ import annotations

from pathlib import Path
import builtins

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


class _Report:
    def __init__(self, tag: str):
        self.tag = tag
        self.messages = [f"{tag}-m1", f"{tag}-m2"]

    def summarize(self):  # noqa: D401
        return f"{self.tag}-summary"


def test_run_train_meta_propagation_copy_and_noops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Prepare directories
    ds = tmp_path / "dataset"
    out = tmp_path / "out"
    ds.mkdir()
    out.mkdir()

    # Base config
    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=ds),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=out),
    )

    class _Trainer:
        def train(self, _cfg):  # noqa: D401
            return _Report("train")

    monkeypatch.setattr(cli._experiment_loader, "load_trainer", lambda _exp: _Trainer())

    # Case 1: source exists, dest missing -> copy occurs
    src = ds / "meta.pkl"
    dst = out / "meta.pkl"
    src.write_bytes(b"SRC")
    if dst.exists():
        dst.unlink()
    cli._run_train("exp", tcfg, tmp_path / "cfg.toml")
    assert dst.exists() and dst.read_bytes() == b"SRC"

    # Case 2: destination exists -> no overwrite
    dst.write_bytes(b"DST")
    # Remove source to be sure copy would fail if attempted
    if src.exists():
        src.unlink()
    cli._run_train("exp", tcfg, tmp_path / "cfg.toml")
    assert dst.read_bytes() == b"DST"

    # Case 3: no source and no dest -> no crash, still missing
    if dst.exists():
        dst.unlink()
    assert not src.exists() and not dst.exists()
    cli._run_train("exp", tcfg, tmp_path / "cfg.toml")
    assert not dst.exists()


def test_run_loop_calls_in_order_and_handles_print_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[str] = []

    class _Prep:
        def prepare(self, cfg):  # noqa: D401
            calls.append("prepare")
            return _Report("prepare")

    class _Trainer:
        def train(self, cfg):  # noqa: D401
            calls.append("train")
            return _Report("train")

    class _Sampler:
        def sample(self, cfg):  # noqa: D401
            calls.append("sample")
            return _Report("sample")

    monkeypatch.setattr(cli._experiment_loader, "load_preparer", lambda _e: _Prep())
    monkeypatch.setattr(cli._experiment_loader, "load_trainer", lambda _e: _Trainer())
    monkeypatch.setattr(cli._experiment_loader, "load_sampler", lambda _e: _Sampler())

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

    # Make all prints raise to exercise print guards
    monkeypatch.setattr(
        builtins, "print", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    # Should not raise
    cli._run_loop("exp", prep_cfg, tcfg, scfg, tmp_path / "cfg.toml")

    assert calls == ["prepare", "train", "sample"]

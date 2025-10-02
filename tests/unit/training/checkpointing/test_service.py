from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from ml_playground.configuration.models import (
    DataConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    SharedConfig,
    TrainerConfig,
    READ_POLICY_BEST,
    READ_POLICY_LATEST,
)
from ml_playground.core.error_handling import CheckpointError
from ml_playground.training.checkpointing import service


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)


class _StubOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0}]

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        del set_to_none


class _StubLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _StubEMA:
    def __init__(self) -> None:
        self.shadow: dict[str, Any] | None = {}


def _make_cfg(tmp_path: Path, *, read_policy: str = READ_POLICY_BEST) -> TrainerConfig:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return TrainerConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=4, block_size=4, dropout=0.0),
        data=DataConfig(batch_size=2, block_size=4, grad_accum_steps=1),
        optim=OptimConfig(learning_rate=0.01),
        schedule=LRSchedule(
            decay_lr=False, warmup_iters=0, lr_decay_iters=0, min_lr=0.0
        ),
        runtime=RuntimeConfig(
            out_dir=out_dir,
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=False,
            seed=1,
            device="cpu",
            dtype="float32",
            compile=False,
            tensorboard_enabled=False,
            ema_decay=0.0,
            checkpointing=RuntimeConfig.Checkpointing(read_policy=read_policy),
        ),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )


def _make_shared(tmp_path: Path, cfg: TrainerConfig) -> SharedConfig:
    return SharedConfig(
        experiment="unit",
        config_path=tmp_path / "cfg.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path / "data",
        train_out_dir=cfg.runtime.out_dir,
        sample_out_dir=cfg.runtime.out_dir,
    )


def _with_checkpoint_load_fn(cfg: TrainerConfig, fn) -> TrainerConfig:
    return cfg.model_copy(update={"checkpoint_load_fn": fn})


def _with_sample_out_dir(shared: SharedConfig, sample_out_dir: Path) -> SharedConfig:
    return shared.model_copy(update={"sample_out_dir": sample_out_dir})


def test_save_checkpoint_invokes_manager(tmp_path: Path) -> None:
    cfg_latest = _make_cfg(tmp_path, read_policy=READ_POLICY_LATEST)
    shared = _make_shared(tmp_path, cfg_latest)
    model = _StubModel()
    optimizer = _StubOptimizer()

    calls: list[dict[str, Any]] = []

    class _Manager:
        def __init__(self) -> None:
            self.out_dir = shared.train_out_dir

        def save_checkpoint(
            self, checkpoint, base_filename, metric, iter_num, logger, is_best
        ):
            calls.append(
                {
                    "metric": metric,
                    "iter_num": iter_num,
                    "is_best": is_best,
                    "model": checkpoint.model,
                }
            )
            return tmp_path / "ckpt.pt"

    mgr = _Manager()
    service.save_checkpoint(
        mgr,
        cfg_latest,
        model=model,
        optimizer=optimizer,
        ema=None,
        iter_num=1,
        best_val_loss=0.123,
        logger=None,
        is_best=True,
    )

    assert calls
    payload = calls[0]
    assert payload["metric"] == pytest.approx(0.123)
    assert payload["iter_num"] == 1
    assert payload["is_best"] is True


def test_load_checkpoint_respects_policy(monkeypatch, tmp_path: Path) -> None:
    cfg_latest = _make_cfg(tmp_path, read_policy=READ_POLICY_LATEST)
    shared = _make_shared(tmp_path, cfg_latest)

    class _Manager:
        def __init__(self) -> None:
            self.out_dir = shared.train_out_dir
            self.best_called = False
            self.last_called = False

        def load_best_checkpoint(self, *, device, logger):
            del device, logger
            self.best_called = True
            return "best"

        def load_latest_checkpoint(self, *, device, logger):
            del device, logger
            self.last_called = True
            return "latest"

    mgr = _Manager()
    result = service.load_checkpoint(mgr, cfg_latest, logger=None)
    assert result == "latest"
    assert mgr.last_called is True

    cfg_best = _make_cfg(tmp_path, read_policy=READ_POLICY_BEST)
    result = service.load_checkpoint(mgr, cfg_best, logger=None)
    assert result == "best"
    assert mgr.best_called is True


def test_load_checkpoint_override_exception(tmp_path: Path) -> None:
    cfg = _with_checkpoint_load_fn(
        _make_cfg(tmp_path),
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    shared = _make_shared(tmp_path, cfg)
    logger = _StubLogger()

    class _Manager:
        def __init__(self) -> None:
            self.out_dir = shared.train_out_dir

    result = service.load_checkpoint(_Manager(), cfg, logger=logger)
    assert result is None
    assert logger.warnings == ["checkpoint_load_fn failed: boom"]


def test_load_checkpoint_missing_out_dir(tmp_path: Path) -> None:
    cfg = _with_checkpoint_load_fn(_make_cfg(tmp_path), None)
    logger = _StubLogger()

    class _Manager:
        def __init__(self) -> None:
            self.out_dir = tmp_path / "missing"

    result = service.load_checkpoint(_Manager(), cfg, logger=logger)
    assert result is None
    assert not logger.warnings


def test_load_checkpoint_handles_checkpoint_error(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, read_policy=READ_POLICY_LATEST)
    logger = _StubLogger()

    class _Manager:
        def __init__(self) -> None:
            self.out_dir = tmp_path / "out_err"
            self.out_dir.mkdir(parents=True, exist_ok=True)

        def load_latest_checkpoint(self, *, device, logger):  # type: ignore[no-untyped-def]
            del device, logger
            raise CheckpointError("bad checkpoint")

    result = service.load_checkpoint(_Manager(), cfg, logger=logger)
    assert result is None
    assert logger.warnings == ["Could not load checkpoint (latest): bad checkpoint"]


def test_propagate_metadata_copies_file(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    shared = _make_shared(tmp_path, cfg)
    ds_dir = shared.dataset_dir
    ds_dir.mkdir(parents=True, exist_ok=True)
    meta_src = ds_dir / "meta.pkl"
    meta_src.write_bytes(b"meta")

    shared = _with_sample_out_dir(shared, tmp_path / "sample-out")

    expanded_shared = shared
    meta_dst = expanded_shared.train_out_dir / meta_src.name

    service.propagate_metadata(cfg, expanded_shared, logger=None)

    assert meta_dst.exists()
    assert meta_dst.read_bytes() == b"meta"

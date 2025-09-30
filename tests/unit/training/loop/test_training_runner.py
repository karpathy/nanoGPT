from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

import ml_playground.training.loop.runner as runner_mod
from ml_playground.configuration.models import (
    DataConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    SharedConfig,
    TrainerConfig,
)
from ml_playground.checkpoint import Checkpoint


class _FakeBatches:
    def __init__(self, device: str) -> None:
        self.device = device

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        del split
        X = torch.zeros((2, 2), device=self.device)
        Y = torch.zeros((2, 2), device=self.device)
        return X, Y


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):  # type: ignore[override]
        del Y
        loss = torch.ones((), device=X.device, requires_grad=True)
        return self.lin(X), loss

    def estimate_mfu(self, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        return 42.0


class _FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0}]

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        del state

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        del set_to_none


class _FakeScaler:
    class _Scaled:
        def backward(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def scale(self, loss: torch.Tensor) -> "_FakeScaler._Scaled":
        del loss
        return _FakeScaler._Scaled()

    def unscale_(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def step(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _FakeWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def close(self) -> None:
        pass


@dataclass
class _Saved:
    is_best: bool
    iter_num: int


class _FakeCkptMgr:
    def __init__(self) -> None:
        self.saved: list[_Saved] = []
        self.out_dir = Path("/tmp")


def _make_cfg(
    tmp_path: Path, *, eval_only: bool = False, max_iters: int = 2
) -> TrainerConfig:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return TrainerConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, dropout=0.0),
        data=DataConfig(batch_size=2, block_size=4, grad_accum_steps=1),
        optim=OptimConfig(
            learning_rate=0.01,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            grad_clip=0.0,
        ),
        schedule=LRSchedule(
            decay_lr=True,
            warmup_iters=1,
            lr_decay_iters=10,
            min_lr=0.001,
        ),
        runtime=RuntimeConfig(
            out_dir=out_dir,
            max_iters=max_iters,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=eval_only,
            seed=42,
            device="cpu",
            dtype="float32",
            compile=False,
            tensorboard_enabled=True,
            ema_decay=0.0,
        ),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )


def _shared(tmp_path: Path, cfg: TrainerConfig) -> SharedConfig:
    return SharedConfig(
        experiment="unit",
        config_path=tmp_path / "cfg.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path / "data",
        train_out_dir=cfg.runtime.out_dir,
        sample_out_dir=cfg.runtime.out_dir,
    )


def test_train_eval_only_breaks_early_and_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner_mod,
        "initialize_batches",
        lambda cfg, shared: _FakeBatches(device="cpu"),
    )

    def _init_model(
        cfg: TrainerConfig, logger: Any
    ) -> Tuple[_FakeModel, _FakeOptimizer]:
        del cfg, logger
        return _FakeModel(), _FakeOptimizer()

    monkeypatch.setattr(runner_mod, "initialize_model", _init_model)
    monkeypatch.setattr(
        runner_mod,
        "initialize_components",
        lambda model, cfg, runtime, log_dir: (
            model,
            _FakeScaler(),
            None,
            _FakeWriter(),
        ),
    )
    monkeypatch.setattr(
        runner_mod, "create_manager", lambda cfg, shared: _FakeCkptMgr()
    )
    monkeypatch.setattr(runner_mod, "load_checkpoint", lambda mgr, cfg, logger: None)
    monkeypatch.setattr(
        runner_mod, "propagate_metadata", lambda cfg, shared, logger: None
    )
    monkeypatch.setattr(
        runner_mod,
        "run_evaluation",
        lambda cfg, logger, iter_num, lr, raw_model, batches, ctx, writer: {
            "train": 0.5,
            "val": 0.4,
        },
    )

    saved_calls: list[Dict[str, Any]] = []

    def _save_checkpoint(
        manager, cfg, *, model, optimizer, ema, iter_num, best_val_loss, logger, is_best
    ) -> None:
        saved_calls.append(
            {
                "iter_num": iter_num,
                "best": is_best,
                "best_val_loss": best_val_loss,
            }
        )

    monkeypatch.setattr(runner_mod, "save_checkpoint", _save_checkpoint)

    cfg = _make_cfg(tmp_path, eval_only=True, max_iters=0)
    shared = _shared(tmp_path, cfg)
    it, best = runner_mod.Trainer(cfg, shared).run()

    assert it == 0
    assert best == pytest.approx(0.4)
    assert any(not call["best"] for call in saved_calls)


def test_train_writes_best_checkpoint_on_improvement_after_first_iter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner_mod,
        "initialize_batches",
        lambda cfg, shared: _FakeBatches(device="cpu"),
    )
    monkeypatch.setattr(
        runner_mod,
        "initialize_model",
        lambda cfg, logger: (_FakeModel(), _FakeOptimizer()),
    )
    monkeypatch.setattr(
        runner_mod,
        "initialize_components",
        lambda model, cfg, runtime, log_dir: (
            model,
            _FakeScaler(),
            None,
            _FakeWriter(),
        ),
    )
    fake_mgr = _FakeCkptMgr()
    monkeypatch.setattr(runner_mod, "create_manager", lambda cfg, shared: fake_mgr)
    monkeypatch.setattr(runner_mod, "load_checkpoint", lambda mgr, cfg, logger: None)
    monkeypatch.setattr(
        runner_mod, "propagate_metadata", lambda cfg, shared, logger: None
    )

    calls: Dict[int, Dict[str, float]] = {
        0: {"train": 0.6, "val": 0.5},
        1: {"train": 0.5, "val": 0.2},
    }

    def _eval(cfg, logger, iter_num, lr, raw_model, batches, ctx, writer):
        del cfg, logger, lr, raw_model, batches, ctx, writer
        return calls.get(iter_num, calls[1])

    monkeypatch.setattr(runner_mod, "run_evaluation", _eval)

    saved_calls: list[_Saved] = []

    def _save_checkpoint(
        manager, cfg, *, model, optimizer, ema, iter_num, best_val_loss, logger, is_best
    ) -> None:
        del manager, cfg, model, optimizer, ema, best_val_loss, logger
        saved_calls.append(_Saved(is_best=is_best, iter_num=iter_num))

    monkeypatch.setattr(runner_mod, "save_checkpoint", _save_checkpoint)

    cfg = _make_cfg(tmp_path, eval_only=False, max_iters=2)
    shared = _shared(tmp_path, cfg)

    it, best = runner_mod.Trainer(cfg, shared).run()

    assert it >= 1
    assert any(call.is_best and call.iter_num == 1 for call in saved_calls)
    assert any(not call.is_best for call in saved_calls)
    assert best == pytest.approx(0.2)


def test_trainer_updates_optimizer_lr_via_get_lr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner_mod,
        "initialize_batches",
        lambda cfg, shared: _FakeBatches(device="cpu"),
    )
    monkeypatch.setattr(
        runner_mod,
        "initialize_model",
        lambda cfg, logger: (_FakeModel(), _FakeOptimizer()),
    )
    monkeypatch.setattr(
        runner_mod,
        "initialize_components",
        lambda model, cfg, runtime, log_dir: (
            model,
            _FakeScaler(),
            None,
            _FakeWriter(),
        ),
    )
    monkeypatch.setattr(
        runner_mod, "create_manager", lambda cfg, shared: _FakeCkptMgr()
    )
    monkeypatch.setattr(runner_mod, "load_checkpoint", lambda mgr, cfg, logger: None)
    monkeypatch.setattr(
        runner_mod, "propagate_metadata", lambda cfg, shared, logger: None
    )
    monkeypatch.setattr(
        runner_mod,
        "run_evaluation",
        lambda cfg, logger, iter_num, lr, raw_model, batches, ctx, writer: {
            "train": 0.5,
            "val": 0.4,
        },
    )
    monkeypatch.setattr(runner_mod, "save_checkpoint", lambda *args, **kwargs: None)

    cfg = _make_cfg(tmp_path, eval_only=False, max_iters=3)
    shared = _shared(tmp_path, cfg)

    lr_calls: list[Tuple[int, float]] = []
    original_get_lr = runner_mod.get_lr

    def _get_lr(it: int, schedule: LRSchedule, optim: OptimConfig) -> float:
        val = original_get_lr(it, schedule, optim)
        lr_calls.append((it, val))
        return val

    monkeypatch.setattr(runner_mod, "get_lr", _get_lr)

    trainer = runner_mod.Trainer(cfg, shared)
    trainer.run()

    assert lr_calls, "get_lr should be invoked"
    for _, lr in lr_calls:
        assert 0.0 <= lr <= cfg.optim.learning_rate + 1e-6
    assert any(lr > 0 for _, lr in lr_calls)
    for group in trainer.optimizer.param_groups:
        assert 0.0 <= group["lr"] <= cfg.optim.learning_rate + 1e-6


def test_get_lr_variants() -> None:
    schedule = LRSchedule(decay_lr=False)
    optim = OptimConfig(learning_rate=0.1)
    assert runner_mod.get_lr(0, schedule, optim) == 0.1

    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    assert runner_mod.get_lr(5, schedule, optim) == pytest.approx(0.05)
    assert runner_mod.get_lr(10, schedule, optim) == 0.1
    assert runner_mod.get_lr(20, schedule, optim) == 0.01
    assert runner_mod.get_lr(25, schedule, optim) == 0.01


def test_checkpoint_model_args() -> None:
    checkpoint_data: Dict[str, Any] = {
        "model_args": {"n_layer": 1},
        "config": {"model_args": {"n_layer": 2}},
        "optimizer": {},
        "iter_num": 0,
        "best_val_loss": 0.0,
        "model": {},
    }
    ckpt = Checkpoint(**checkpoint_data)
    assert ckpt.model_args == {"n_layer": 1}

    checkpoint_data = {
        "config": {"model_args": {"n_layer": 2}},
        "optimizer": {},
        "iter_num": 0,
        "best_val_loss": 0.0,
        "model": {},
    }
    model_args = checkpoint_data.get("model_args") or checkpoint_data["config"].get(
        "model_args"
    )
    checkpoint_data["model_args"] = model_args
    ckpt = Checkpoint(**checkpoint_data)
    assert ckpt.model_args == {"n_layer": 2}

    checkpoint_data = {
        "config": {},
        "optimizer": {},
        "iter_num": 0,
        "best_val_loss": 0.0,
        "model": {},
    }
    with pytest.raises(TypeError):
        Checkpoint(**checkpoint_data)

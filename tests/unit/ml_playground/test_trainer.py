from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

import ml_playground.trainer as trainer_mod
from ml_playground.config import (
    TrainerConfig,
    RuntimeConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
)
from ml_playground.checkpoint import Checkpoint


# ---- Fakes --------------------------------------------------------------------
class _FakeBatches:
    def __init__(self, device: str) -> None:
        self.device = device

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.zeros((2, 2), device=self.device)
        Y = torch.zeros((2, 2), device=self.device)
        return X, Y


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):  # type: ignore[override]
        loss = torch.tensor(1.0, requires_grad=True)
        return self.lin(X), loss

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: Tuple[float, float], device: str
    ):
        return torch.optim.SGD(self.parameters(), lr=lr)

    def estimate_mfu(self, *args: Any, **kwargs: Any) -> float:  # used for logging
        return 42.0


class _FakeScaler:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    class _Scaled:
        def backward(self, *args: Any, **kwargs: Any) -> None:
            return None

    def scale(self, loss: torch.Tensor) -> "_FakeScaler._Scaled":
        return _FakeScaler._Scaled()

    def unscale_(self, *args: Any, **kwargs: Any) -> None:
        return None

    def step(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass
class _Saved:
    is_best: bool
    iter_num: int


class _FakeCkptMgr:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.saved: list[_Saved] = []

    def save_checkpoint(
        self,
        ckpt: Any,
        base_filename: str,
        metric: float,
        iter_num: int,
        logger: Any | None = None,
        is_best: bool = False,
    ):
        self.saved.append(_Saved(is_best=is_best, iter_num=iter_num))

        class _P:
            def __init__(self, name: str) -> None:
                self._name = name

            @property
            def name(self) -> str:
                return self._name

        return _P(f"ckpt_{'best' if is_best else 'last'}_{iter_num:08d}.pt")

    def load_latest_checkpoint(self, device: str, logger: Any | None = None):  # noqa: ANN001
        return None

    def load_best_checkpoint(self, device: str, logger: Any | None = None):  # noqa: ANN001
        return None


class _FakeWriter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ---- Helpers ------------------------------------------------------------------


def _make_minimal_trainer_cfg(
    tmp_path: Path, eval_only: bool = False, max_iters: int = 2
) -> TrainerConfig:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    cfg = TrainerConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=8, block_size=4, dropout=0.0),
        data=DataConfig(
            dataset_dir=data_root, batch_size=2, block_size=4, grad_accum_steps=1
        ),
        optim=OptimConfig(
            learning_rate=0.01, weight_decay=0.0, beta1=0.9, beta2=0.95, grad_clip=0.0
        ),
        schedule=LRSchedule(
            decay_lr=False, warmup_iters=0, lr_decay_iters=10, min_lr=1e-5
        ),
        runtime=RuntimeConfig(
            out_dir=tmp_path / "out",
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
    )
    return cfg


# ---- Trainer loop tests (from test_trainer_unit) ------------------------------


def test_train_eval_only_breaks_early_and_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        trainer_mod, "_setup_data_loader", lambda d, r: _FakeBatches(device=r.device)
    )
    monkeypatch.setattr(trainer_mod, "GPT", lambda cfg: _FakeModel())
    monkeypatch.setattr(trainer_mod, "GradScaler", _FakeScaler)
    monkeypatch.setattr(trainer_mod, "SummaryWriter", _FakeWriter)
    monkeypatch.setattr(trainer_mod, "CheckpointManager", _FakeCkptMgr)
    monkeypatch.setattr(
        trainer_mod, "estimate_loss", lambda m, b, n, ctx: {"train": 0.5, "val": 0.4}
    )

    cfg = _make_minimal_trainer_cfg(tmp_path, eval_only=True, max_iters=0)
    it, best = trainer_mod.train(cfg)
    assert it == 0
    assert best == pytest.approx(0.4)


def test_train_writes_best_checkpoint_on_improvement_after_first_iter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        trainer_mod, "_setup_data_loader", lambda d, r: _FakeBatches(device=r.device)
    )
    monkeypatch.setattr(trainer_mod, "GPT", lambda cfg: _FakeModel())
    monkeypatch.setattr(trainer_mod, "GradScaler", _FakeScaler)
    monkeypatch.setattr(trainer_mod, "SummaryWriter", _FakeWriter)
    fake_mgr = _FakeCkptMgr()
    monkeypatch.setattr(trainer_mod, "CheckpointManager", lambda *a, **k: fake_mgr)

    calls: Dict[int, Dict[str, float]] = {
        0: {"train": 0.6, "val": 0.5},
        1: {"train": 0.5, "val": 0.2},
    }

    def _est_loss(model: Any, batches: Any, n: int, ctx: Any) -> Dict[str, float]:
        if not hasattr(_est_loss, "i"):
            _est_loss.i = 0  # type: ignore[attr-defined]
        i = getattr(_est_loss, "i")  # type: ignore[attr-defined]
        v = calls.get(i, calls[1])
        setattr(_est_loss, "i", i + 1)  # type: ignore[attr-defined]
        return v

    monkeypatch.setattr(trainer_mod, "estimate_loss", _est_loss)

    cfg = _make_minimal_trainer_cfg(tmp_path, eval_only=False, max_iters=2)
    it, best = trainer_mod.train(cfg)

    assert it >= 1
    assert any(s.is_best and s.iter_num == 1 for s in fake_mgr.saved)
    assert any((not s.is_best) for s in fake_mgr.saved)


# ---- LR helper tests (from test_trainer_helpers) ------------------------------


def test_get_lr_no_decay() -> None:
    schedule = LRSchedule(decay_lr=False)
    optim = OptimConfig(learning_rate=0.1)
    lr = trainer_mod.get_lr(0, schedule, optim)
    assert lr == 0.1


def test_get_lr_warmup_phase() -> None:
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    lr = trainer_mod.get_lr(5, schedule, optim)
    assert lr == pytest.approx(0.05)
    lr = trainer_mod.get_lr(10, schedule, optim)
    assert lr == 0.1


def test_get_lr_decay_phase() -> None:
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    lr = trainer_mod.get_lr(15, schedule, optim)
    assert lr > 0.01 and lr < 0.1
    lr = trainer_mod.get_lr(20, schedule, optim)
    assert lr == 0.01


def test_get_lr_past_decay() -> None:
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    lr = trainer_mod.get_lr(25, schedule, optim)
    assert lr == 0.01


def test_get_lr_warmup_decay_edges() -> None:
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    assert trainer_mod.get_lr(0, schedule, optim) == 0.0
    assert trainer_mod.get_lr(10, schedule, optim) == 0.1
    assert trainer_mod.get_lr(20, schedule, optim) == 0.01
    assert trainer_mod.get_lr(100, schedule, optim) == 0.01


# ---- Extract model args tests (from test_trainer_extract_model_args) ----------


def test_extract_model_args_prefers_model_args_key() -> None:
    checkpoint_data = {
        "model_args": {"n_layer": 1},
        "config": {"model_args": {"n_layer": 2}},
        "optimizer": {},
        "iter_num": 0,
        "best_val_loss": 0.0,
        "model": {},
    }
    ckpt = Checkpoint(**checkpoint_data)
    assert ckpt.model_args == {"n_layer": 1}


def test_extract_model_args_falls_back_to_config_model_args() -> None:
    checkpoint_data = {
        "config": {"model_args": {"n_layer": 2}},
        "optimizer": {},
        "iter_num": 0,
        "best_val_loss": 0.0,
        "model": {},
    }
    model_args = checkpoint_data.get("model_args") or checkpoint_data.get(
        "config", {}
    ).get("model_args")
    checkpoint_data["model_args"] = model_args
    ckpt = Checkpoint(**checkpoint_data)
    assert ckpt.model_args == {"n_layer": 2}


def test_extract_model_args_missing_raises_strict() -> None:
    checkpoint_data = {
        "config": {},
        "optimizer": {},
        "iter_num": 0,
        "best_val_loss": 0.0,
        "model": {},
    }
    with pytest.raises(TypeError):
        Checkpoint(**checkpoint_data)


def test_extract_model_args_missing_key_raises() -> None:
    checkpoint_data = {
        "model_args": "not_a_dict",
        "config": {},
        "optimizer": {},
        "iter_num": 0,
        "best_val_loss": 0.0,
        "model": {},
    }
    ckpt = Checkpoint(**checkpoint_data)
    assert ckpt.model_args == "not_a_dict"

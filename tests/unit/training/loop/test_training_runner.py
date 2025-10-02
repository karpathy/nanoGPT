from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pytest
import torch

import ml_playground.training.loop.runner as runner_mod
from ml_playground.training.loop.runner import TrainerDependencies
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


def _build_deps(
    *,
    evaluation: Dict[int, Dict[str, float]] | None = None,
    saved_hook: Callable[[Dict[str, Any]], None] | None = None,
    load_checkpoint_result: Optional[Checkpoint] = None,
    get_lr_override: Callable[[int, LRSchedule, OptimConfig], float] | None = None,
) -> Tuple[TrainerDependencies, _FakeCkptMgr]:
    evaluation = evaluation or {}
    batches = _FakeBatches(device="cpu")
    model = _FakeModel()
    optimizer = _FakeOptimizer()
    scaler = _FakeScaler()
    writer = _FakeWriter()
    manager = _FakeCkptMgr()

    def init_batches(cfg: TrainerConfig, shared: SharedConfig) -> _FakeBatches:
        del cfg, shared
        return batches

    def init_model(
        cfg: TrainerConfig, logger: Any
    ) -> Tuple[_FakeModel, _FakeOptimizer]:
        del cfg, logger
        return model, optimizer

    def init_components(
        model_param: torch.nn.Module,
        cfg: TrainerConfig,
        runtime: runner_mod.RuntimeContext,
        log_dir: str,
    ) -> Tuple[torch.nn.Module, _FakeScaler, None, _FakeWriter]:
        del cfg, runtime, log_dir
        return model_param, scaler, None, writer

    def create_manager(cfg: TrainerConfig, shared: SharedConfig) -> _FakeCkptMgr:
        del cfg, shared
        return manager

    def load_checkpoint(
        manager_param: _FakeCkptMgr,
        cfg: TrainerConfig,
        *,
        logger: Any,
    ) -> Optional[Checkpoint]:
        del manager_param, cfg, logger
        return load_checkpoint_result

    def apply_checkpoint(
        checkpoint: Checkpoint,
        *,
        model: torch.nn.Module,
        optimizer: Any,
        ema: Optional[Any],
    ) -> Tuple[int, float]:
        del model, optimizer, ema
        return checkpoint.iter_num, checkpoint.best_val_loss

    def save_checkpoint(
        manager_param: _FakeCkptMgr,
        cfg: TrainerConfig,
        *,
        model: torch.nn.Module,
        optimizer: Any,
        ema: Optional[Any],
        iter_num: int,
        best_val_loss: float,
        logger: Any,
        is_best: bool,
    ) -> None:
        del cfg, model, optimizer, ema, logger
        manager_param.saved.append(_Saved(is_best=is_best, iter_num=iter_num))
        if saved_hook is not None:
            saved_hook(
                {
                    "iter_num": iter_num,
                    "best": is_best,
                    "best_val_loss": best_val_loss,
                }
            )

    def propagate_metadata(
        cfg: TrainerConfig,
        shared: SharedConfig,
        *,
        logger: Any,
    ) -> None:
        del cfg, shared, logger

    def run_evaluation(
        cfg: TrainerConfig,
        *,
        logger: Any,
        iter_num: int,
        lr: float,
        raw_model: torch.nn.Module,
        batches: Any,
        ctx: Any,
        writer: Optional[_FakeWriter],
    ) -> Dict[str, float]:
        del cfg, logger, lr, raw_model, batches, ctx, writer
        return evaluation.get(iter_num, evaluation.get(-1, {"train": 0.5, "val": 0.5}))

    def get_lr(iteration: int, schedule: LRSchedule, optim: OptimConfig) -> float:
        if get_lr_override is not None:
            return get_lr_override(iteration, schedule, optim)
        return runner_mod.get_lr(iteration, schedule, optim)

    deps = TrainerDependencies(
        initialize_batches=init_batches,
        initialize_model=init_model,
        initialize_components=init_components,
        create_manager=create_manager,
        load_checkpoint=load_checkpoint,
        apply_checkpoint=apply_checkpoint,
        save_checkpoint=save_checkpoint,
        propagate_metadata=propagate_metadata,
        run_evaluation=run_evaluation,
        get_lr=get_lr,
    )
    return deps, manager


def test_train_eval_only_breaks_early_and_returns(tmp_path: Path) -> None:
    saved_calls: list[Dict[str, Any]] = []
    deps, _manager = _build_deps(
        evaluation={0: {"train": 0.5, "val": 0.4}},
        saved_hook=saved_calls.append,
    )

    cfg = _make_cfg(tmp_path, eval_only=True, max_iters=0)
    shared = _shared(tmp_path, cfg)

    it, best = runner_mod.Trainer(cfg, shared, deps=deps).run()

    assert it == 0
    assert best == pytest.approx(0.4)
    assert any(not call["best"] for call in saved_calls)


def test_train_writes_best_checkpoint_on_improvement_after_first_iter(
    tmp_path: Path,
) -> None:
    calls: Dict[int, Dict[str, float]] = {
        0: {"train": 0.6, "val": 0.5},
        1: {"train": 0.5, "val": 0.2},
    }
    saved_calls: list[Dict[str, Any]] = []
    deps, manager = _build_deps(
        evaluation=calls,
        saved_hook=saved_calls.append,
    )

    cfg = _make_cfg(tmp_path, eval_only=False, max_iters=2)
    shared = _shared(tmp_path, cfg)

    it, best = runner_mod.Trainer(cfg, shared, deps=deps).run()

    assert it >= 1
    assert any(call["best"] and call["iter_num"] == 1 for call in saved_calls)
    assert any(not call["best"] for call in saved_calls)
    assert best == pytest.approx(0.2)
    assert manager.saved, "checkpoints should be recorded"


def test_trainer_updates_optimizer_lr_via_get_lr(tmp_path: Path) -> None:
    lr_calls: list[Tuple[int, float]] = []

    def track_lr(iteration: int, schedule: LRSchedule, optim: OptimConfig) -> float:
        value = runner_mod.get_lr(iteration, schedule, optim)
        lr_calls.append((iteration, value))
        return value

    deps, _manager = _build_deps(
        evaluation={0: {"train": 0.5, "val": 0.4}},
        get_lr_override=track_lr,
    )

    cfg = _make_cfg(tmp_path, eval_only=False, max_iters=3)
    shared = _shared(tmp_path, cfg)

    trainer = runner_mod.Trainer(cfg, shared, deps=deps)
    trainer.run()

    assert lr_calls, "get_lr must be invoked at least once"
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

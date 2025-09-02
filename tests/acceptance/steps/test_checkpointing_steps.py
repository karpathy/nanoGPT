from __future__ import annotations
from pathlib import Path
import time

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from ml_playground.trainer import Checkpoint, CheckpointManager


scenarios("../features/checkpointing.feature")


@pytest.fixture
def tmp_ckpt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dummy_checkpoint(iter_num: int, best_val: float) -> Checkpoint:
    return Checkpoint(
        model={"w": 1},
        optimizer={"o": 1},
        model_args={"n_layer": 1, "n_head": 1, "n_embd": 4, "block_size": 8, "bias": False, "vocab_size": 16, "dropout": 0.0},
        iter_num=iter_num,
        best_val_loss=best_val,
        config={"ok": True},
        ema=None,
    )


@given("a fresh checkpoints directory")
def fresh_dir(tmp_ckpt_dir: Path) -> Path:
    return tmp_ckpt_dir


@given(parsers.parse("a checkpoint manager with keep_last {keep_last:d} and keep_best {keep_best:d}"), target_fixture="manager")
def _manager_fixture(tmp_ckpt_dir: Path, keep_last: int, keep_best: int) -> CheckpointManager:
    return CheckpointManager(tmp_ckpt_dir, atomic=True, keep_last=keep_last, keep_best=keep_best)


@when("I save 3 last checkpoints sequentially")
def save_three_last(manager: CheckpointManager):
    for i in range(3):
        ckpt = _dummy_checkpoint(i, 1e9)
        manager.save_checkpoint(ckpt, "ckpt_last.pt", metric=1e9, iter_num=i, logger=None, is_best=False)
        time.sleep(0.01)


@then("only the 2 most recent last checkpoints exist")
def assert_two_last(manager: CheckpointManager):
    files = sorted(p.name for p in manager.out_dir.glob("ckpt_last_*.pt"))
    assert len(files) == 2
    # last two iters: 1 and 2
    assert files[-2].startswith("ckpt_last_00000001")
    assert files[-1].startswith("ckpt_last_00000002")


@then("a stable last pointer exists")
def stable_last_pointer(manager: CheckpointManager):
    p = manager.out_dir / "ckpt_last.pt"
    assert p.exists()


@when(parsers.parse("I save 3 best checkpoints with metrics {m1:f}, {m2:f}, {m3:f}"))
def save_three_best(manager: CheckpointManager, m1: float, m2: float, m3: float):
    for i, m in enumerate([m1, m2, m3]):
        ckpt = _dummy_checkpoint(i, m)
        manager.save_checkpoint(ckpt, "ckpt_best.pt", metric=m, iter_num=i, logger=None, is_best=True)
        time.sleep(0.01)


@then("only the 2 best checkpoints by metric exist")
def assert_two_best(manager: CheckpointManager):
    files = list(sorted(manager.out_dir.glob("ckpt_best_*.pt")))
    # keep_best=2 -> keep two with lowest metric: 0.9 and 1.0
    names = [p.name for p in files]
    assert any("_00000001_0.900000.pt" in n or n.endswith("_0.900000.pt") for n in names)
    assert any("_00000000_1.000000.pt" in n or n.endswith("_1.000000.pt") for n in names)
    assert not any("_1.100000.pt" in n for n in names)


@then("a stable best pointer exists")
def stable_best_pointer(manager: CheckpointManager):
    p = manager.out_dir / "ckpt_best.pt"
    assert p.exists()


@when("I save 2 last checkpoints sequentially")
def save_two_last(manager: CheckpointManager):
    for i in range(2):
        ckpt = _dummy_checkpoint(i, 1e9)
        manager.save_checkpoint(ckpt, "ckpt_last.pt", metric=1e9, iter_num=i, logger=None, is_best=False)
        time.sleep(0.01)


@when("I reinitialize the checkpoint manager", target_fixture="reinit_manager")
def _reinit_manager_fixture(manager: CheckpointManager) -> CheckpointManager:
    # Recreate to force discovery
    return CheckpointManager(manager.out_dir, atomic=True, keep_last=manager.keep_last, keep_best=manager.keep_best)


@then("the manager discovers the existing last checkpoints")
def discovered_last(reinit_manager: CheckpointManager):
    assert len(reinit_manager.last_checkpoints) >= 2


@when("I simulate an eval step with an improvement at iter 0")
def simulate_first_improvement(manager: CheckpointManager):
    ckpt = _dummy_checkpoint(0, 0.5)
    manager.save_checkpoint(ckpt, "ckpt_best.pt", metric=0.5, iter_num=0, logger=None, is_best=True)
    manager.save_checkpoint(ckpt, "ckpt_last.pt", metric=0.5, iter_num=0, logger=None, is_best=False)


@then("both best and last checkpoints exist for iter 0")
def both_exist_iter0(manager: CheckpointManager):
    assert any(p.name.startswith("ckpt_best_00000000") for p in manager.out_dir.glob("ckpt_best_*.pt"))
    assert any(p.name.startswith("ckpt_last_00000000") for p in manager.out_dir.glob("ckpt_last_*.pt"))


@when("I simulate an eval step with an improvement at iter 10")
def simulate_improvement_iter10(manager: CheckpointManager):
    ckpt = _dummy_checkpoint(10, 0.4)
    manager.save_checkpoint(ckpt, "ckpt_best.pt", metric=0.4, iter_num=10, logger=None, is_best=True)


@then("only a best checkpoint exists for iter 10 (no last in same step)")
def only_best_iter10(manager: CheckpointManager):
    assert any(p.name.startswith("ckpt_best_00000010") for p in manager.out_dir.glob("ckpt_best_*.pt"))
    assert not any(p.name.startswith("ckpt_last_00000010") for p in manager.out_dir.glob("ckpt_last_*.pt"))

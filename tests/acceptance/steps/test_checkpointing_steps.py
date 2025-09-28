from __future__ import annotations
from pathlib import Path
import time

import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from ml_playground.checkpoint import Checkpoint, CheckpointManager


scenarios("../features/checkpointing.feature")

# Constants
_TEST_LOGGER = __import__("logging").getLogger("test")
_MODEL_ARGS = {
    "n_layer": 1,
    "n_head": 1,
    "n_embd": 4,
    "block_size": 8,
    "bias": False,
    "vocab_size": 16,
    "dropout": 0.0,
}


@pytest.fixture
def tmp_ckpt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dummy_checkpoint(iter_num: int, best_val: float) -> Checkpoint:
    return Checkpoint(
        model={"w": 1},
        optimizer={"o": 1},
        model_args=_MODEL_ARGS,
        iter_num=iter_num,
        best_val_loss=best_val,
        config={"ok": True},
        ema=None,
    )


def _save_checkpoint(
    manager: CheckpointManager,
    ckpt: Checkpoint,
    filename: str,
    metric: float,
    iter_num: int,
    is_best: bool,
):
    """Helper to save a checkpoint with consistent parameters."""
    manager.save_checkpoint(
        ckpt,
        filename,
        metric=metric,
        iter_num=iter_num,
        logger=_TEST_LOGGER,
        is_best=is_best,
    )


@given("a fresh checkpoints directory")
def fresh_dir(tmp_ckpt_dir: Path) -> Path:
    return tmp_ckpt_dir


@given(
    parsers.parse(
        "checkpoint retention policy of {keep_last:d} last, {keep_best:d} best",
    ),
    target_fixture="manager",
)
def checkpoint_retention_policy(
    tmp_ckpt_dir: Path, keep_last: int, keep_best: int
) -> CheckpointManager:
    assert tmp_ckpt_dir is not None, "tmp_ckpt_dir fixture required"
    assert keep_last >= 0, f"keep_last must be non-negative, got {keep_last}"
    assert keep_best >= 0, f"keep_best must be non-negative, got {keep_best}"
    return CheckpointManager(
        tmp_ckpt_dir, atomic=True, keep_last=keep_last, keep_best=keep_best
    )


@when(parsers.parse("{count:d} checkpoints are saved sequentially"))
def save_checkpoints_sequentially(manager: CheckpointManager, count: int):
    assert manager is not None, "CheckpointManager required"
    assert count > 0, f"count must be positive, got {count}"
    for i in range(count):
        ckpt = _dummy_checkpoint(i, 1e9)
        _save_checkpoint(manager, ckpt, "ckpt_last.pt", 1e9, i, False)
        time.sleep(0.01)


@when("checkpoints are saved with the following metrics:")
def save_checkpoints_with_metrics(manager: CheckpointManager, datatable):
    assert manager is not None, "CheckpointManager required"
    assert datatable, "Expected datatable with metrics data"
    assert len(datatable) >= 2, "Expected header row + data rows in datatable"

    headers = datatable[0]
    rows = datatable[1:]
    metrics_table = [dict(zip(headers, row)) for row in rows]

    for i, row in enumerate(metrics_table):
        metric = float(row["metric"])
        ckpt = _dummy_checkpoint(i, metric)
        _save_checkpoint(manager, ckpt, "ckpt_best.pt", metric, i, True)
        time.sleep(0.01)


@when("the checkpoint manager is reinitialized", target_fixture="reinit_manager")
def reinitialize_checkpoint_manager(manager: CheckpointManager) -> CheckpointManager:
    assert manager is not None, "CheckpointManager required"
    assert manager.out_dir is not None, "Manager must have out_dir"
    # Recreate to force discovery
    return CheckpointManager(
        manager.out_dir,
        atomic=True,
        keep_last=manager.keep_last,
        keep_best=manager.keep_best,
    )


@when(
    parsers.parse("an evaluation step produces improvement at iteration {iter_num:d}")
)
def simulate_evaluation_improvement(manager: CheckpointManager, iter_num: int):
    assert manager is not None, "CheckpointManager required"
    assert iter_num >= 0, f"iter_num must be non-negative, got {iter_num}"
    metric = 0.5 if iter_num == 0 else 0.4
    ckpt = _dummy_checkpoint(iter_num, metric)
    _save_checkpoint(manager, ckpt, "ckpt_best.pt", metric, iter_num, True)
    # Only save last checkpoint if it's the first improvement
    if iter_num == 0:
        _save_checkpoint(manager, ckpt, "ckpt_last.pt", metric, iter_num, False)


def _checkpoint_exists(manager: CheckpointManager, pattern: str, iter_num: int) -> bool:
    """Check if a checkpoint file exists for the given iteration."""
    assert manager is not None, "CheckpointManager required"
    assert pattern, "pattern required"
    assert iter_num >= 0, f"iter_num must be non-negative, got {iter_num}"
    return any(
        p.name.startswith(f"{pattern}_{iter_num:08d}")
        for p in manager.out_dir.glob(f"{pattern}_*.pt")
    )


@then(parsers.parse("{count:d} most recent checkpoints should exist"))
def assert_most_recent_checkpoints_exist(manager: CheckpointManager, count: int):
    assert manager is not None, "CheckpointManager required"
    assert count > 0, f"count must be positive, got {count}"
    files = sorted(p.name for p in manager.out_dir.glob("ckpt_last_*.pt"))
    assert len(files) == count
    # Verify the most recent checkpoints exist (hardcoded for test scenario)
    for i in range(count):
        expected_iter = i + (3 - count)  # Based on test saving 3 checkpoints
        expected_pattern = f"ckpt_last_{expected_iter:08d}"
        assert any(expected_pattern in f for f in files), (
            f"Expected {expected_pattern} in {files}"
        )


@then(parsers.parse("{count:d} best checkpoints by metric should exist"))
def assert_best_checkpoints_by_metric_exist(manager: CheckpointManager, count: int):
    assert manager is not None, "CheckpointManager required"
    assert count >= 0, f"count must be non-negative, got {count}"
    files = list(manager.out_dir.glob("ckpt_best_*.pt"))
    assert len(files) == count
    names = [p.name for p in files]
    # Verify expected metrics exist (based on test data: 1.0, 0.9, 1.1 → keep 0.9, 1.0)
    assert any("0.900000" in n for n in names), f"Expected 0.9 metric in {names}"
    assert any("1.000000" in n for n in names), f"Expected 1.0 metric in {names}"
    assert not any("1.100000" in n for n in names), f"1.1 should not exist in {names}"


@then("no stable last checkpoint pointer should exist")
def assert_no_stable_last_pointer(manager: CheckpointManager):
    assert manager is not None, "CheckpointManager required"
    assert not (manager.out_dir / "ckpt_last.pt").exists()


@then("no stable best checkpoint pointer should exist")
def assert_no_stable_best_pointer(manager: CheckpointManager):
    assert manager is not None, "CheckpointManager required"
    assert not (manager.out_dir / "ckpt_best.pt").exists()


@then("existing checkpoints should be discovered")
def assert_existing_checkpoints_discovered(reinit_manager: CheckpointManager):
    assert reinit_manager is not None, "Reinitialized CheckpointManager required"
    assert len(reinit_manager.last_checkpoints) >= 2


@then(
    parsers.parse(
        "both best and last checkpoints should exist for iteration {iter_num:d}"
    )
)
def assert_both_checkpoints_exist_for_iteration(
    manager: CheckpointManager, iter_num: int
):
    assert manager is not None, "CheckpointManager required"
    assert iter_num >= 0, f"iter_num must be non-negative, got {iter_num}"
    assert _checkpoint_exists(manager, "ckpt_best", iter_num)
    assert _checkpoint_exists(manager, "ckpt_last", iter_num)


@then(parsers.parse("only best checkpoint should exist for iteration {iter_num:d}"))
def assert_only_best_checkpoint_exists_for_iteration(
    manager: CheckpointManager, iter_num: int
):
    assert manager is not None, "CheckpointManager required"
    assert iter_num >= 0, f"iter_num must be non-negative, got {iter_num}"
    assert _checkpoint_exists(manager, "ckpt_best", iter_num)
    assert not _checkpoint_exists(manager, "ckpt_last", iter_num)

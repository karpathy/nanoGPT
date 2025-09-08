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
        model_args={
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 4,
            "block_size": 8,
            "bias": False,
            "vocab_size": 16,
            "dropout": 0.0,
        },
        iter_num=iter_num,
        best_val_loss=best_val,
        config={"ok": True},
        ema=None,
    )


@given("a fresh checkpoints directory")
def fresh_dir(tmp_ckpt_dir: Path) -> Path:
    return tmp_ckpt_dir


@given(
    parsers.cfparse(
        "checkpoint retention policy of {keep_last:d} last, {keep_best:d} best",
        extra_types={"d": int},
    ),
    target_fixture="manager",
)
def checkpoint_retention_policy(
    tmp_ckpt_dir: Path, keep_last: int, keep_best: int
) -> CheckpointManager:
    return CheckpointManager(
        tmp_ckpt_dir, atomic=True, keep_last=keep_last, keep_best=keep_best
    )


@when(
    parsers.cfparse(
        "{count:d} checkpoints are saved sequentially", extra_types={"d": int}
    )
)
def save_checkpoints_sequentially(manager: CheckpointManager, count: int):
    for i in range(count):
        ckpt = _dummy_checkpoint(i, 1e9)
        manager.save_checkpoint(
            ckpt, "ckpt_last.pt", metric=1e9, iter_num=i, logger=None, is_best=False
        )
        time.sleep(0.01)


@when("checkpoints are saved with the following metrics:")
def save_checkpoints_with_metrics(manager: CheckpointManager, datatable):
    """Save checkpoints with the provided metrics from the data table."""
    # Convert datatable to list of dictionaries
    if datatable:
        # Skip the header row and convert each row to a dictionary
        headers = datatable[0]
        rows = datatable[1:]
        metrics_table = [dict(zip(headers, row)) for row in rows]

        for i, row in enumerate(metrics_table):
            metric = float(row["metric"])
            iter_num = i  # Use index as iteration number
            ckpt = _dummy_checkpoint(iter_num, metric)
            manager.save_checkpoint(
                ckpt,
                "ckpt_best.pt",
                metric=metric,
                iter_num=iter_num,
                logger=None,
                is_best=True,
            )
            time.sleep(0.01)


@when("the checkpoint manager is reinitialized", target_fixture="reinit_manager")
def reinitialize_checkpoint_manager(manager: CheckpointManager) -> CheckpointManager:
    # Recreate to force discovery
    return CheckpointManager(
        manager.out_dir,
        atomic=True,
        keep_last=manager.keep_last,
        keep_best=manager.keep_best,
    )


@when(
    parsers.cfparse(
        "an evaluation step produces improvement at iteration {iter_num:d}",
        extra_types={"d": int},
    )
)
def simulate_evaluation_improvement(manager: CheckpointManager, iter_num: int):
    # Use a better metric (lower is better) to simulate improvement
    metric = 0.5 if iter_num == 0 else 0.4
    ckpt = _dummy_checkpoint(iter_num, metric)
    manager.save_checkpoint(
        ckpt,
        "ckpt_best.pt",
        metric=metric,
        iter_num=iter_num,
        logger=None,
        is_best=True,
    )
    # Only save last checkpoint if it's the first improvement
    if iter_num == 0:
        manager.save_checkpoint(
            ckpt,
            "ckpt_last.pt",
            metric=metric,
            iter_num=iter_num,
            logger=None,
            is_best=False,
        )


@then(
    parsers.cfparse(
        "{count:d} most recent checkpoints should exist", extra_types={"d": int}
    )
)
def assert_most_recent_checkpoints_exist(manager: CheckpointManager, count: int):
    files = sorted(p.name for p in manager.out_dir.glob("ckpt_last_*.pt"))
    assert len(files) == count
    # When we save N checkpoints and keep only K, we should have the most recent K
    # For example: save 3, keep 2 → should have checkpoints 1 and 2 (iterations 1 and 2)
    # The files are sorted by name, so the most recent will be at the end
    for i in range(count):
        # We expect the last 'count' files to be the most recent iterations
        # Since we saved 3 checkpoints (0, 1, 2), with keep_last=2, we should have 1 and 2
        expected_iter = i + (3 - count)  # 3 - 2 = 1, so iterations 1 and 2
        expected_filename_part = f"ckpt_last_{expected_iter:08d}"
        assert any(expected_filename_part in f for f in files), (
            f"Expected {expected_filename_part} in files: {files}"
        )


@then(
    parsers.cfparse(
        "{count:d} best checkpoints by metric should exist", extra_types={"d": int}
    )
)
def assert_best_checkpoints_by_metric_exist(manager: CheckpointManager, count: int):
    files = list(sorted(manager.out_dir.glob("ckpt_best_*.pt")))
    # keep_best policy should keep the specified number of best checkpoints
    assert len(files) == count
    # Verify they have the expected metrics (0.9 and 1.0 are better than 1.1)
    names = [p.name for p in files]
    # With keep_best=2, we should keep the 2 best metrics: 0.9 and 1.0
    assert any("0.900000" in n for n in names), (
        f"Expected 0.9 metric checkpoint, found: {names}"
    )
    assert any("1.000000" in n for n in names), (
        f"Expected 1.0 metric checkpoint, found: {names}"
    )
    # 1.1 should NOT be present since it's worse than 0.9 and 1.0
    assert not any("1.100000" in n for n in names), (
        f"1.1 metric checkpoint should not exist, found: {names}"
    )


@then("no stable last checkpoint pointer should exist")
def assert_no_stable_last_pointer(manager: CheckpointManager):
    p = manager.out_dir / "ckpt_last.pt"
    assert not p.exists()


@then("no stable best checkpoint pointer should exist")
def assert_no_stable_best_pointer(manager: CheckpointManager):
    p = manager.out_dir / "ckpt_best.pt"
    assert not p.exists()


@then("existing checkpoints should be discovered")
def assert_existing_checkpoints_discovered(reinit_manager: CheckpointManager):
    assert len(reinit_manager.last_checkpoints) >= 2


@then(
    parsers.cfparse(
        "both best and last checkpoints should exist for iteration {iter_num:d}",
        extra_types={"d": int},
    )
)
def assert_both_checkpoints_exist_for_iteration(
    manager: CheckpointManager, iter_num: int
):
    # Check best checkpoint exists
    assert any(
        p.name.startswith(f"ckpt_best_{iter_num:08d}")
        for p in manager.out_dir.glob("ckpt_best_*.pt")
    )
    # Check last checkpoint exists
    assert any(
        p.name.startswith(f"ckpt_last_{iter_num:08d}")
        for p in manager.out_dir.glob("ckpt_last_*.pt")
    )


@then(
    parsers.cfparse(
        "only best checkpoint should exist for iteration {iter_num:d}",
        extra_types={"d": int},
    )
)
def assert_only_best_checkpoint_exists_for_iteration(
    manager: CheckpointManager, iter_num: int
):
    # Check best checkpoint exists
    assert any(
        p.name.startswith(f"ckpt_best_{iter_num:08d}")
        for p in manager.out_dir.glob("ckpt_best_*.pt")
    )
    # Check last checkpoint does NOT exist for this iteration
    assert not any(
        p.name.startswith(f"ckpt_last_{iter_num:08d}")
        for p in manager.out_dir.glob("ckpt_last_*.pt")
    )

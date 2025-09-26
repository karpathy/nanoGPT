from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from ml_playground.checkpoint import (
    Checkpoint,
    CheckpointManager,
    CheckpointError,
)


# -----------------------------------------------------------------------------
# Shared fixture
# -----------------------------------------------------------------------------


@pytest.fixture()
def ckpt_obj() -> Checkpoint:
    return Checkpoint(
        model={"w": [1, 2, 3]},
        optimizer={"state": {}},
        model_args={"vocab_size": 16},
        iter_num=0,
        best_val_loss=1.0,
        config={"foo": "bar"},
    )


# -----------------------------------------------------------------------------
# CheckpointManager rotation/loading/validation tests
# -----------------------------------------------------------------------------


def test_checkpoint_manager_rotation_and_latest(
    tmp_path: Path, ckpt_obj: Checkpoint, out_dir: Path
) -> None:
    out = out_dir
    mgr = CheckpointManager(out, atomic=False, keep_last=2, keep_best=1)

    # Save 3 last checkpoints -> only last 2 should remain
    for it in range(3):
        mgr.save_checkpoint(
            ckpt_obj,
            base_filename="ignored",
            metric=float("inf"),
            iter_num=it,
            logger=logging.getLogger("test"),
        )

    last_files = sorted(out.glob("ckpt_last_*.pt"))
    # Keep policy respected
    assert len(last_files) == 2
    assert last_files[-1].name.endswith("00000002.pt")

    # Load latest works
    ck = mgr.load_latest_checkpoint(device="cpu", logger=logging.getLogger("test"))
    assert isinstance(ck, Checkpoint)
    assert ck.model_args["vocab_size"] == 16


def test_checkpoint_manager_best_rotation_and_sidecar_cleanup(
    tmp_path: Path, ckpt_obj: Checkpoint, out_dir: Path
) -> None:
    out = out_dir
    logger = logging.getLogger("test")
    mgr = CheckpointManager(out, atomic=False, keep_last=0, keep_best=1)

    # Save worse metric first and create its sidecar
    p_worse = mgr.save_checkpoint(
        ckpt_obj,
        base_filename="ignored",
        metric=2.0,
        iter_num=11,
        is_best=True,
        logger=logger,
    )
    sidecar = p_worse.with_suffix(p_worse.suffix + ".json")
    sidecar.write_text(json.dumps({"info": 1}))
    # Now save the better metric, which should trigger deletion of the worse one and its sidecar
    _p_better = mgr.save_checkpoint(
        ckpt_obj,
        base_filename="ignored",
        metric=1.234567,
        iter_num=10,
        is_best=True,
        logger=logger,
    )

    # Only one best should remain (the better one, metric=1.234567)
    best_files = list(out.glob("ckpt_best_*.pt"))
    assert len(best_files) == 1
    assert "00000010_1.234567" in best_files[0].name
    # Sidecar for the removed (worse) one should not exist
    assert not sidecar.exists()


def test_discovery_and_errors_in_init(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()

    # Create a malformed last checkpoint file name
    bad = out / "ckpt_last_badname.pt"
    torch.save({"x": 1}, bad)

    with pytest.raises(CheckpointError):
        _ = CheckpointManager(out)


def test_load_latest_errors_and_type_validation(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    mgr = CheckpointManager(out, atomic=False)

    # No checkpoints present -> error on load_latest
    with pytest.raises(CheckpointError):
        mgr.load_latest_checkpoint(device="cpu", logger=logging.getLogger("test"))

    # Write a non-dict checkpoint that will be discovered and fail validation
    p = out / "ckpt_last_00000001.pt"
    torch.save("not-a-dict", p)
    # Clear any cached state to force discovery
    mgr.last_checkpoints.clear()
    with pytest.raises(CheckpointError, match="mapping payload"):
        mgr.load_latest_checkpoint(device="cpu", logger=logging.getLogger("test"))


def test_keep_policy_validation() -> None:
    with pytest.raises(CheckpointError):
        _ = CheckpointManager(Path("/tmp/does-not-matter"), keep_last=-1)
    with pytest.raises(CheckpointError):
        _ = CheckpointManager(Path("/tmp/does-not-matter"), keep_best=-2)


# -----------------------------------------------------------------------------
# Checkpoint payload validation
# -----------------------------------------------------------------------------


def test_checkpoint_from_payload_success(ckpt_obj: Checkpoint) -> None:
    payload = ckpt_obj.to_dict()
    restored = Checkpoint.from_payload(payload)
    assert restored.iter_num == ckpt_obj.iter_num
    assert restored.best_val_loss == ckpt_obj.best_val_loss
    assert restored.model["w"] == [1, 2, 3]


def test_checkpoint_from_payload_missing_key(ckpt_obj: Checkpoint) -> None:
    payload = ckpt_obj.to_dict()
    payload.pop("model")  # type: ignore[arg-type]
    with pytest.raises(CheckpointError, match="missing required fields: model"):
        Checkpoint.from_payload(payload)


def test_checkpoint_from_payload_wrong_type(ckpt_obj: Checkpoint) -> None:
    payload = ckpt_obj.to_dict()
    payload["iter_num"] = "oops"  # type: ignore[assignment]
    with pytest.raises(CheckpointError, match="iter_num"):
        Checkpoint.from_payload(payload)


# -----------------------------------------------------------------------------
# Atomic and non-atomic save tests (migrated from test_trainer_utils.py)
# -----------------------------------------------------------------------------


def test_save_checkpoint_atomic_and_readback(tmp_path: Path) -> None:
    # Atomic save via CheckpointManager should create only final file, no lingering .tmp
    mgr = CheckpointManager(out_dir=tmp_path, atomic=True, keep_last=1, keep_best=0)
    ckpt = Checkpoint(
        model={"w": torch.tensor([1, 2])},
        optimizer={"state": {}},
        model_args={
            "vocab_size": 16,
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
        },
        iter_num=1,
        best_val_loss=1.0,
        config={},
    )
    rotated = mgr.save_checkpoint(
        checkpoint=ckpt,
        base_filename="ckpt_last.pt",
        metric=float("inf"),
        iter_num=1,
        logger=logging.getLogger("test"),
        is_best=False,
    )
    assert rotated.exists()
    # No lingering tmp from atomic save
    assert not rotated.with_suffix(rotated.suffix + ".tmp").exists()
    loaded = torch.load(rotated, map_location="cpu")
    assert isinstance(loaded, dict) and "model" in loaded


def test_save_checkpoint_non_atomic(tmp_path: Path) -> None:
    mgr = CheckpointManager(out_dir=tmp_path, atomic=False, keep_last=1, keep_best=0)
    ckpt = Checkpoint(
        model={"w": torch.tensor([3, 4])},
        optimizer={"state": {}},
        model_args={
            "vocab_size": 16,
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
        },
        iter_num=2,
        best_val_loss=1.0,
        config={},
    )
    rotated = mgr.save_checkpoint(
        checkpoint=ckpt,
        base_filename="ckpt_last.pt",
        metric=float("inf"),
        iter_num=2,
        logger=logging.getLogger("test"),
        is_best=False,
    )
    assert rotated.exists()
    loaded = torch.load(rotated, map_location="cpu")
    assert isinstance(loaded, dict) and loaded["iter_num"] == 2

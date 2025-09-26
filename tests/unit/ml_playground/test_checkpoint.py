from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping
from unittest.mock import patch

import pytest
import torch

from ml_playground.checkpoint import (
    Checkpoint,
    CheckpointManager,
    CheckpointError,
    CheckpointLoadError,
    TorchUnpicklingError,
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
    # Verify the logging output
    with patch.object(logger, "info") as mock_info:
        mgr.save_checkpoint(
            ckpt_obj,
            base_filename="ignored",
            metric=1.0,  # A better metric to trigger pruning
            iter_num=12,
            is_best=True,
            logger=logger,
        )
        # Create a new best checkpoint to trigger pruning of the one with metric 1.234567
        mgr.save_checkpoint(
            ckpt_obj,
            base_filename="ignored",
            metric=0.5,
            iter_num=13,
            is_best=True,
            logger=logger,
        )
        # Check that the log message for removing the old checkpoint was called
        # We need to check the calls made to the mock logger
        found_log = any(
            "Removed old best checkpoint" in call.args[0]
            for call in mock_info.call_args_list
        )
        assert found_log, (
            "The log message for removing an old checkpoint was not found."
        )


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


def test_checkpoint_from_payload_best_val_loss_type(ckpt_obj: Checkpoint) -> None:
    """Ensure tuple-type validation path is exercised for best_val_loss."""

    payload = ckpt_obj.to_dict()
    payload["best_val_loss"] = "not-a-number"  # type: ignore[assignment]
    with pytest.raises(CheckpointError, match="best_val_loss"):
        Checkpoint.from_payload(payload)


def test_checkpoint_from_payload_model_mapping_validation(ckpt_obj: Checkpoint) -> None:
    """Checkpoint fields that must be mappings raise descriptive errors."""

    payload = ckpt_obj.to_dict()
    payload["model"] = [1, 2, 3]  # type: ignore[assignment]
    with pytest.raises(CheckpointError, match="must be a mapping"):
        Checkpoint.from_payload(payload)


def test_checkpoint_from_payload_handles_optional_ema(ckpt_obj: Checkpoint) -> None:
    """`ema` payloads should be preserved when reconstructing checkpoints."""

    payload = ckpt_obj.to_dict()
    payload["ema"] = {"decay": 0.95}
    restored = Checkpoint.from_payload(payload)
    assert restored.ema == {"decay": 0.95}


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


def test_checkpoint_to_dict_includes_optional_ema() -> None:
    """Optional EMA data should be present when serializing checkpoints."""

    ckpt = Checkpoint(
        model={"w": torch.tensor([1])},
        optimizer={"state": {}},
        model_args={"n_layer": 1},
        iter_num=3,
        best_val_loss=0.3,
        config={"foo": "bar"},
        ema={"beta": 0.9},
    )
    serialized = ckpt.to_dict()
    assert serialized["ema"] == {"beta": 0.9}


def test_load_latest_checkpoint_wraps_runtime_errors(
    tmp_path: Path, ckpt_obj: Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_latest_checkpoint should wrap torch.load failures in CheckpointError."""

    logger = logging.getLogger("test")
    mgr = CheckpointManager(tmp_path, atomic=False, keep_last=1, keep_best=0)
    mgr.save_checkpoint(
        ckpt_obj,
        base_filename="ignored",
        metric=float("inf"),
        iter_num=1,
        logger=logger,
    )

    def _boom(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        raise RuntimeError("torch-load-failure")

    monkeypatch.setattr("ml_playground.checkpoint.torch.load", _boom)

    called: dict[str, object] = {}

    def fake_add_safe_globals(globs: list[object]) -> None:
        called["globs"] = globs

    monkeypatch.setattr(
        "ml_playground.checkpoint.torch.serialization.add_safe_globals",
        fake_add_safe_globals,
    )

    with pytest.raises(CheckpointError, match="Failed to load checkpoint"):
        mgr.load_latest_checkpoint(device="cpu", logger=logger)

    assert "globs" in called


def test_load_best_checkpoint_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best checkpoint loading should succeed and return a reconstructed object."""

    logger = logging.getLogger("test")
    mgr = CheckpointManager(tmp_path, atomic=False, keep_last=0, keep_best=1)
    ckpt = Checkpoint(
        model={"w": torch.tensor([5])},
        optimizer={"state": {}},
        model_args={"n_layer": 2},
        iter_num=5,
        best_val_loss=0.2,
        config={"bar": 1},
    )
    mgr.save_checkpoint(
        ckpt,
        base_filename="ignored",
        metric=0.2,
        iter_num=5,
        logger=logger,
        is_best=True,
    )

    monkeypatch.setattr(
        "ml_playground.checkpoint.torch.serialization.add_safe_globals",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("duplicate")),
    )

    monkeypatch.setattr(
        "ml_playground.checkpoint.torch.load",
        lambda *_args, **_kwargs: ckpt.to_dict(),
    )

    with patch.object(logger, "info") as mock_info:
        loaded = mgr.load_best_checkpoint(device="cpu", logger=logger)
        assert loaded.iter_num == ckpt.iter_num
        assert loaded.best_val_loss == ckpt.best_val_loss
        mock_info.assert_called_with(
            f"Loaded checkpoint from {mgr.best_checkpoints[0].path}"
        )


def test_discover_existing_best_invalid_metric(tmp_path: Path) -> None:
    """Malformed best-checkpoint filenames should raise informative errors."""

    out = tmp_path / "out"
    out.mkdir()
    bad_path = out / "ckpt_best_00000010_notanumber.pt"
    torch.save({"dummy": 1}, bad_path)

    with pytest.raises(CheckpointError, match="Could not parse metric"):
        CheckpointManager(out, keep_last=0, keep_best=1)


def test_discover_existing_last_malformed_and_stat_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed filenames and stat errors are surfaced during discovery."""

    out = tmp_path / "out"
    out.mkdir()

    malformed = out / "ckpt_last_bad.pt"
    torch.save({"dummy": 0}, malformed)

    with pytest.raises(CheckpointError, match="Could not parse iteration"):
        CheckpointManager(out, keep_last=1, keep_best=0)

    malformed = out / "ckpt_last_notanint.pt"
    torch.save({"dummy": 1}, malformed)

    with pytest.raises(CheckpointError, match="Could not parse iteration"):
        CheckpointManager(out, keep_last=1, keep_best=0)

    good_path = out / "ckpt_last_00000001.pt"
    torch.save({"ok": 1}, good_path)

    original_stat = Path.stat

    def boom_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:  # type: ignore[override]
        if self == good_path:
            raise OSError("stat failed")
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", boom_stat, raising=False)

    with pytest.raises(CheckpointError, match="Failed to stat checkpoint"):
        CheckpointManager(out, keep_last=1, keep_best=0)


def test_discover_existing_best_filename_errors(tmp_path: Path) -> None:
    """Best checkpoint discovery surfaces malformed names and parsing errors."""

    out = tmp_path / "out"
    out.mkdir()

    malformed = out / "ckpt_best_bad.pt"
    torch.save({"x": 1}, malformed)

    with pytest.raises(CheckpointError, match="Could not parse iteration"):
        CheckpointManager(out, keep_last=0, keep_best=1)

    bad_iter = out / "ckpt_best_prefix_notanint_1.0.pt"
    torch.save({"x": 1}, bad_iter)

    with pytest.raises(CheckpointError, match="Could not parse iteration"):
        CheckpointManager(out, keep_last=0, keep_best=1)

    bad_metric = out / "ckpt_best_00000001_notafloat.pt"
    torch.save({"x": 1}, bad_metric)

    with pytest.raises(CheckpointError, match="Could not parse metric"):
        CheckpointManager(out, keep_last=0, keep_best=1)


def test_discover_existing_best_populates_entries(tmp_path: Path) -> None:
    """Existing best checkpoints should be loaded into manager state."""

    out = tmp_path / "out"
    out.mkdir()
    best = out / "ckpt_best_00000002_1.234000.pt"
    torch.save({"model": {}}, best)

    mgr = CheckpointManager(out, keep_last=0, keep_best=2)
    assert len(mgr.best_checkpoints) == 1
    assert mgr.best_checkpoints[0].path == best


def test_save_checkpoint_prune_last_failure(
    tmp_path: Path, ckpt_obj: Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors during last-checkpoint pruning should raise `CheckpointError`."""

    logger = logging.getLogger("test_prune_last")
    mgr = CheckpointManager(tmp_path, atomic=False, keep_last=1, keep_best=0)
    mgr.save_checkpoint(
        ckpt_obj,
        base_filename="ignored",
        metric=float("inf"),
        iter_num=0,
        logger=logger,
    )

    original_unlink = Path.unlink

    def boom_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith("ckpt_last_00000000"):
            raise OSError("can't remove")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", boom_unlink, raising=False)

    with pytest.raises(CheckpointError, match="Failed to remove old last checkpoint"):
        mgr.save_checkpoint(
            ckpt_obj,
            base_filename="ignored",
            metric=float("inf"),
            iter_num=1,
            logger=logger,
        )


def test_save_checkpoint_prune_best_failure(
    tmp_path: Path, ckpt_obj: Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors during best-checkpoint pruning should raise `CheckpointError`."""

    logger = logging.getLogger("test_prune_best")
    mgr = CheckpointManager(tmp_path, atomic=False, keep_last=0, keep_best=1)
    mgr.save_checkpoint(
        ckpt_obj,
        base_filename="ignored",
        metric=2.0,
        iter_num=1,
        logger=logger,
        is_best=True,
    )

    original_unlink = Path.unlink

    def boom_unlink_best(self: Path, missing_ok: bool = False) -> None:
        if "1.000000" in self.name or "2.000000" in self.name:
            raise OSError("can't remove best")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", boom_unlink_best, raising=False)

    with pytest.raises(CheckpointError, match="Failed to remove old best checkpoint"):
        mgr.save_checkpoint(
            ckpt_obj,
            base_filename="ignored",
            metric=1.0,
            iter_num=2,
            logger=logger,
            is_best=True,
        )


def test_load_latest_checkpoint_wraps_unpickling_error(
    tmp_path: Path, ckpt_obj: Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Torch unpickling errors should be wrapped in `CheckpointError`."""

    logger = logging.getLogger("test_unpickle_latest")
    mgr = CheckpointManager(tmp_path, atomic=False, keep_last=1, keep_best=0)
    mgr.save_checkpoint(
        ckpt_obj,
        base_filename="ignored",
        metric=float("inf"),
        iter_num=3,
        logger=logger,
    )

    monkeypatch.setattr(
        "ml_playground.checkpoint.torch.load",
        lambda *_a, **_kw: (_ for _ in ()).throw(TorchUnpicklingError("bad pickle")),
    )

    with pytest.raises(CheckpointError, match="Failed to load checkpoint"):
        mgr.load_latest_checkpoint(device="cpu", logger=logger)


def test_load_best_checkpoint_no_existing(tmp_path: Path) -> None:
    """Attempting to load best checkpoint with none available raises error."""

    logger = logging.getLogger("test_best_missing")
    mgr = CheckpointManager(tmp_path, atomic=False, keep_last=0, keep_best=1)
    with pytest.raises(CheckpointError, match="No best checkpoints"):
        mgr.load_best_checkpoint(device="cpu", logger=logger)


def test_discover_best_checkpoint_no_metric(tmp_path: Path) -> None:
    """Best checkpoints without a metric in the filename should be discovered."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "ckpt_best_00000001.pt").touch()
    mgr = CheckpointManager(out, keep_last=0, keep_best=1)
    assert len(mgr.best_checkpoints) == 1
    assert mgr.best_checkpoints[0].metric == float("inf")


def test_safe_globals_branches(
    tmp_path: Path, ckpt_obj: Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover branches where add_safe_globals or its dependencies are missing."""
    out = tmp_path / "out"
    out.mkdir()
    mgr = CheckpointManager(out, keep_last=1, keep_best=0)
    mgr.save_checkpoint(ckpt_obj, "ignored", 1.0, 1, logging.getLogger("test"))

    # Case 1: add_safe_globals is not callable
    monkeypatch.setattr("torch.serialization.add_safe_globals", None)
    mgr.load_latest_checkpoint("cpu", logging.getLogger("test"))

    # Case 2: PosixPath class is not found
    monkeypatch.setattr("torch.serialization.add_safe_globals", lambda *a, **kw: None)
    monkeypatch.setattr("pathlib._local", None)
    mgr.load_latest_checkpoint("cpu", logging.getLogger("test"))


def test_add_safe_globals_exception_path(
    tmp_path: Path, ckpt_obj: Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the exception handling in add_safe_globals is covered."""
    out = tmp_path / "out"
    out.mkdir()
    mgr = CheckpointManager(out, keep_last=1, keep_best=0)
    mgr.save_checkpoint(ckpt_obj, "ignored", 1.0, 1, logging.getLogger("test"))

    def boom_add_safe_globals(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Boom!")

    monkeypatch.setattr("torch.serialization.add_safe_globals", boom_add_safe_globals)
    # The call should not raise an exception
    mgr.load_latest_checkpoint("cpu", logging.getLogger("test"))


def test_load_best_checkpoint_final_coverage(
    tmp_path: Path, ckpt_obj: Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover the final remaining branches in load_best_checkpoint."""
    out = tmp_path / "out"
    out.mkdir()
    mgr = CheckpointManager(out, keep_last=0, keep_best=1)

    # 1. Test the "no best checkpoints" error path
    with pytest.raises(CheckpointError, match="No best checkpoints discovered"):
        mgr.load_best_checkpoint("cpu", logging.getLogger("test"))

    # 2. Save a checkpoint to test the load paths
    mgr.save_checkpoint(
        ckpt_obj,
        "ignored",
        metric=1.0,
        iter_num=1,
        logger=logging.getLogger("test"),
        is_best=True,
    )

    # 3. Test the CheckpointLoadError path
    monkeypatch.setattr(
        "ml_playground.checkpoint.torch.load",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(CheckpointLoadError):
        mgr.load_best_checkpoint("cpu", logging.getLogger("test"))

    # 4. Test the normal load path to cover add_safe_globals
    monkeypatch.undo()
    mgr.load_best_checkpoint("cpu", logging.getLogger("test"))


def test_discover_existing_best_stat_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stat failure on a best checkpoint should raise CheckpointError."""
    out = tmp_path / "out"
    out.mkdir()
    best_path = out / "ckpt_best_00000001_1.0.pt"
    torch.save({}, best_path)

    original_stat = Path.stat

    def boom_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self == best_path:
            raise OSError("stat failed")
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", boom_stat, raising=False)

    with pytest.raises(CheckpointError, match="Failed to stat checkpoint"):
        CheckpointManager(out, keep_last=0, keep_best=1)


def test_load_best_checkpoint_discovers_from_disk(
    tmp_path: Path, ckpt_obj: Checkpoint
) -> None:
    """If no best checkpoints are in memory, discovery should run."""
    out = tmp_path / "out"
    out.mkdir()
    mgr = CheckpointManager(out, keep_last=0, keep_best=1)
    mgr.save_checkpoint(
        ckpt_obj,
        base_filename="ignored",
        metric=1.0,
        iter_num=1,
        logger=logging.getLogger("test"),
        is_best=True,
    )
    # Create a new manager to simulate a restart where memory is empty
    mgr_new = CheckpointManager(out, keep_last=0, keep_best=1)
    assert mgr_new.best_checkpoints
    loaded_ckpt = mgr_new.load_best_checkpoint("cpu", logger=logging.getLogger("test"))
    assert loaded_ckpt.iter_num == ckpt_obj.iter_num


def test_checkpoint_payload_not_a_mapping(tmp_path: Path, ckpt_obj: Checkpoint) -> None:
    """Loading a checkpoint file that is not a dict should fail."""
    out = tmp_path / "out"
    out.mkdir()
    mgr = CheckpointManager(out, keep_last=1, keep_best=0)
    p = mgr.save_checkpoint(
        ckpt_obj,
        "ignored",
        metric=1.0,
        iter_num=1,
        logger=logging.getLogger("test"),
    )
    # Overwrite with invalid content
    torch.save("not a dict", p)

    with pytest.raises(CheckpointError, match="does not contain a mapping"):
        mgr.load_latest_checkpoint("cpu", logger=logging.getLogger("test"))

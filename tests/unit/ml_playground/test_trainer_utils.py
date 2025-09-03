from __future__ import annotations
from pathlib import Path
import torch

from ml_playground.checkpoint import Checkpoint, CheckpointManager


def test_save_checkpoint_atomic_and_readback(tmp_path: Path):
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
        logger=None,
        is_best=False,
    )
    assert rotated.exists()
    # No lingering tmp from atomic save
    assert not rotated.with_suffix(rotated.suffix + ".tmp").exists()
    loaded = torch.load(rotated, map_location="cpu")
    assert isinstance(loaded, dict) and "model" in loaded


def test_save_checkpoint_non_atomic(tmp_path: Path):
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
        logger=None,
        is_best=False,
    )
    assert rotated.exists()
    loaded = torch.load(rotated, map_location="cpu")
    assert isinstance(loaded, dict) and loaded["iter_num"] == 2

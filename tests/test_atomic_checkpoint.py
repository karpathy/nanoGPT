"""Checkpoint saves should never replace a usable checkpoint with partial data."""

import ast
import os
import tempfile
from pathlib import Path

import pytest
import torch


def _save_fn():
    train = Path(__file__).resolve().parents[1] / "train.py"
    source = ast.parse(train.read_text(encoding="utf-8"), filename=str(train))
    fn = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint_atomic"
    )
    namespace = {"os": os, "tempfile": tempfile, "torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(train), "exec"), namespace)
    return namespace["save_checkpoint_atomic"]


def test_atomic_checkpoint_replaces_previous_checkpoint(tmp_path):
    save = _save_fn()
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save({"step": 1}, checkpoint_path)

    save({"step": 2, "weights": torch.arange(4)}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["step"] == 2
    assert torch.equal(checkpoint["weights"], torch.arange(4))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.pt"]


def test_failed_checkpoint_save_preserves_last_good_file(tmp_path, monkeypatch):
    save = _save_fn()
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save({"step": 1}, checkpoint_path)
    previous_bytes = checkpoint_path.read_bytes()

    def fail_after_partial_write(checkpoint, path):
        Path(path).write_bytes(b"not a valid checkpoint")
        raise OSError("simulated full disk")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated full disk"):
        save({"step": 2}, checkpoint_path)

    assert checkpoint_path.read_bytes() == previous_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.pt"]

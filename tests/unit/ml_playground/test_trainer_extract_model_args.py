from __future__ import annotations

import pytest
import ml_playground.trainer as trainer


def test_extract_model_args_prefers_model_args_key():
    ckpt = {"model_args": {"block_size": 8, "vocab_size": 100}}
    assert trainer.extract_model_args_from_checkpoint(ckpt) == {
        "block_size": 8,
        "vocab_size": 100,
    }


def test_extract_model_args_missing_raises_strict():
    ckpt = {"config": {"model": {"block_size": 16, "vocab_size": 200}}}
    with pytest.raises(trainer.CheckpointError, match="model_args"):
        trainer.extract_model_args_from_checkpoint(ckpt)


def test_extract_model_args_missing_key_raises():
    ckpt = {"something_else": 1}
    with pytest.raises(trainer.CheckpointError, match="model_args"):
        trainer.extract_model_args_from_checkpoint(ckpt)

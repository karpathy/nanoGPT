from __future__ import annotations

from pathlib import Path
import pickle
import torch
import pytest

from ml_playground.config import SamplerConfig, SampleConfig, RuntimeConfig, RuntimeConfig as RC
from ml_playground.model import GPTConfig, GPT
from ml_playground.sampler import sample
from ml_playground.error_handling import DataError, CheckpointError


def _make_minimal_model() -> GPT:
    conf = GPTConfig(
        n_layer=1,
        n_head=1,
        n_embd=32,
        block_size=16,
        bias=False,
        vocab_size=256,
        dropout=0.0,
    )
    return GPT(conf)


def _rotated_best(out_dir: Path, model: GPT) -> Path:
    p = out_dir / "ckpt_best_00000000_0.000000.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {},
            "model_args": {
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 32,
                "block_size": 16,
                "bias": False,
                "vocab_size": 256,
                "dropout": 0.0,
            },
            "iter_num": 0,
            "best_val_loss": 0.0,
            "config": {},
        },
        p,
    )
    return p


def _sampler_cfg(out_dir: Path) -> SamplerConfig:
    return SamplerConfig(
        runtime=RuntimeConfig(
            out_dir=out_dir,
            max_iters=0,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=False,
            checkpointing=RC.Checkpointing(
                keep=RC.Checkpointing.Keep(last=1, best=1)
            ),
            seed=123,
            device="cpu",
            dtype="float32",
            compile=False,
        ),
        sample=SampleConfig(start="\n", num_samples=1, max_new_tokens=1, temperature=1.0, top_k=10),
    )


def test_setup_tokenizer_requires_tokenizer_type(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # valid rotated checkpoint so we reach tokenizer stage
    model = _make_minimal_model()
    _rotated_best(out_dir, model)
    # meta without tokenizer_type
    meta = {
        "meta_version": 1,
        "kind": "char",
        "dtype": "uint16",
        "stoi": {chr(i): i for i in range(256)},
        "itos": {i: chr(i) for i in range(256)},
    }
    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    cfg = _sampler_cfg(out_dir)
    with pytest.raises(DataError):
        sample(cfg)


def test_sampler_requires_rotated_checkpoints(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Provide strict meta so tokenizer would succeed, but omit rotated checkpoints
    meta = {
        "meta_version": 1,
        "kind": "char",
        "tokenizer_type": "char",
        "dtype": "uint16",
        "stoi": {chr(i): i for i in range(256)},
        "itos": {i: chr(i) for i in range(256)},
    }
    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    # Write only stable file (should be ignored) to prove strictness
    torch.save({"model": {}}, out_dir / "ckpt_best.pt")

    cfg = _sampler_cfg(out_dir)
    with pytest.raises(CheckpointError):
        sample(cfg)

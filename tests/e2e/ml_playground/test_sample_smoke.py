from __future__ import annotations
from pathlib import Path
import torch
import logging
from ml_playground.models.core.config import GPTConfig
from ml_playground.models.core.model import GPT
from ml_playground.sampling.runner import Sampler
from ml_playground.configuration.models import (
    SamplerConfig,
    SampleConfig,
    RuntimeConfig,
    RuntimeConfig as RC,
    READ_POLICY_BEST,
    SharedConfig,
)


def test_sample_smoke(tmp_path: Path) -> None:
    # fabricate a tiny model checkpoint
    conf = GPTConfig(
        n_layer=1,
        n_head=1,
        n_embd=32,
        block_size=16,
        bias=False,
        vocab_size=256,
        dropout=0.0,
    )
    model = GPT(conf, logging.getLogger(__name__))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # write strict meta required by sampler
    import pickle

    stoi = {chr(i): i for i in range(256)}
    itos = {i: chr(i) for i in range(256)}
    meta = {
        "meta_version": 1,
        "kind": "char",
        "tokenizer_type": "char",
        "dtype": "uint16",
        "stoi": stoi,
        "itos": itos,
    }
    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    # Write a rotated best checkpoint to satisfy strict loader
    ckpt_best_rotated = out_dir / "ckpt_best_00000000_0.000000.pt"
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
        ckpt_best_rotated,
    )

    exp = SamplerConfig(
        runtime=RuntimeConfig(
            out_dir=out_dir,
            max_iters=0,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=False,
            checkpointing=RC.Checkpointing(
                keep=RC.Checkpointing.Keep(
                    last=1,
                    best=1,
                ),
                read_policy=READ_POLICY_BEST,
            ),
            seed=123,
            device="cpu",
            dtype="float32",
            compile=False,
        ),
        sample=SampleConfig(
            start="\n", num_samples=1, max_new_tokens=4, temperature=1.0, top_k=10
        ),
    )
    shared = SharedConfig(
        experiment="smoke",
        config_path=out_dir / "cfg.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path,  # not used in sampler, but required
        train_out_dir=out_dir,
        sample_out_dir=out_dir,
    )
    Sampler(exp, shared).run()  # should run without exceptions

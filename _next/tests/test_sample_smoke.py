from __future__ import annotations
from pathlib import Path
import torch
from _next.model import GPTConfig, GPT
from _next.sampler import sample
from _next.config import SampleExperiment, SampleConfig, RuntimeConfig


def test_sample_smoke(tmp_path: Path) -> None:
    # fabricate a tiny model checkpoint
    conf = GPTConfig(n_layer=1, n_head=1, n_embd=32, block_size=16, bias=False, vocab_size=256, dropout=0.0)
    model = GPT(conf)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    torch.save({
        "model": model.state_dict(),
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
    }, out_dir / "ckpt_best.pt")

    exp = SampleExperiment(
        runtime=RuntimeConfig(out_dir=out_dir, max_iters=0, eval_interval=1, eval_iters=1, log_interval=1,
                              eval_only=False, always_save_checkpoint=False, seed=123, device="cpu", dtype="float32", compile=False),
        sample=SampleConfig(start="\n", num_samples=1, max_new_tokens=4, temperature=1.0, top_k=10),
    )
    sample(exp)  # should run without exceptions

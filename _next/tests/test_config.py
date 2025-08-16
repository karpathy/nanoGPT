from __future__ import annotations
from pathlib import Path
from _next.config import load_toml, AppConfig


def test_load_toml_roundtrip(tmp_path: Path) -> None:
    toml_text = (
        """
[train.model]
n_layer=1
n_head=1
n_embd=32
block_size=16
bias=false

[train.data]
dataset_dir = "data/shakespeare"
block_size = 16
batch_size = 2
grad_accum_steps = 1

[train.optim]
learning_rate = 0.001

[train.schedule]

[train.runtime]
out_dir = "out/test_next"
max_iters = 1

[sample.runtime]
out_dir = "out/test_next"

[sample.sample]
"""
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text)

    cfg: AppConfig = load_toml(cfg_path)
    assert cfg.train is not None
    assert cfg.sample is not None
    assert isinstance(cfg.train.runtime.out_dir, Path)
    assert isinstance(cfg.train.data.dataset_dir, Path)

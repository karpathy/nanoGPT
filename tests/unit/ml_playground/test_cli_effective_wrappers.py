from __future__ import annotations

from pathlib import Path
import ml_playground.cli as cli


def test_load_train_config_resolves_relative_paths(tmp_path: Path):
    # Create a dummy default config in the parent of the experiments dir
    default_config_path = tmp_path / "default_config.toml"
    default_config_path.write_text("""
[train.model]
 n_layer = 1
 n_head = 1
 n_embd = 32
 block_size = 16
[train.optim]
 learning_rate = 0.001
[train.schedule]
""")

    exp_dir = tmp_path / "experiments" / "exp"
    exp_dir.mkdir(parents=True)
    config_path = exp_dir / "config.toml"
    toml_text = """
[train.data]
 dataset_dir = "data_rel"

[train.runtime]
 out_dir = "out_rel"
"""
    config_path.write_text(toml_text)

    # Use strict loader with explicit path so we don't depend on package experiments root
    _, cfg = cli.def_load_effective_train("exp", config_path)

    assert str(cfg.data.dataset_dir).startswith(str(exp_dir))
    assert str(cfg.runtime.out_dir).startswith(str(exp_dir))


def test_load_sample_config_resolves_relative_out_dir_with_runtime_ref(tmp_path: Path):
    # Use runtime_ref to defaults' train.runtime and override out_dir relatively
    p = tmp_path / "experiments" / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """[sample]
runtime_ref = "train.runtime"

[train]
[train.runtime]
device = "cpu"
"""
    )
    default_config_p = tmp_path / "default_config.toml"
    default_config_p.write_text(
        """[train]
[train.runtime]
out_dir = "out_rel"
"""
    )

    # Use strict loader with explicit path; will resolve runtime_ref and make out_dir absolute
    _, cfg = cli.def_load_effective_sample("exp", p)
    assert cfg.runtime.out_dir.is_absolute()
    assert str(cfg.runtime.out_dir).startswith(str(p.parent))

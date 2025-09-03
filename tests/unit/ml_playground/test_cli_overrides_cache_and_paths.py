from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    pass
else:
    pass

import pytest
import ml_playground.cli as cli


def test_get_cfg_path_explicit_and_default(tmp_path: Path):
    explicit = tmp_path / "x.toml"
    # Explicit path returned as-is
    p = cli._cfg_path_for("some_exp", explicit)
    assert p == explicit
    # Default path under experiments root
    d = cli._cfg_path_for("bundestag_char", None)
    assert d.as_posix().endswith("ml_playground/experiments/bundestag_char/config.toml")


def test_load_config_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1) Missing config path -> exit
    with pytest.raises(FileNotFoundError):
        cli.def_load_effective_train("exp", Path("/__no_such_file__"))

    # Create experiments structure: tmp/experiments/exp/config.toml
    exp_dir = tmp_path / "experiments" / "exp"
    exp_dir.mkdir(parents=True)
    cfg = exp_dir / "config.toml"
    cfg.write_text(
        "[train]\n[train.runtime]\nout_dir='.'\n[train.model]\n[train.data]\ndataset_dir='.'\n[train.optim]\n[train.schedule]"
    )

    # 2) Defaults invalid -> exit mentioning defaults path sibling to experiments/
    defaults_path = tmp_path / "default_config.toml"
    defaults_path.write_text("this is not valid toml")

    with pytest.raises(Exception) as ei3:
        cli.def_load_effective_train("exp", cfg)
    assert "default_config.toml" in str(ei3.value).lower()

    # 3) Experiment config invalid -> exit mentioning cfg path
    bad_cfg = exp_dir / "bad.toml"
    bad_cfg.write_text("this is not valid toml")
    with pytest.raises(Exception) as ei4:
        cli.def_load_effective_train("exp", bad_cfg)
    assert "bad.toml" in str(ei4.value).lower()

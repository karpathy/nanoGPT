from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import logging

import pytest
import typer

import ml_playground.cli as cli
import ml_playground.config_loader as config_loader
from ml_playground.prepare import PreparerConfig
from ml_playground.config import (
    TrainerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
)


def test_experiment_loader_import_failure_exits(monkeypatch: pytest.MonkeyPatch):
    loader = cli.ExperimentLoader()
    # Force import error
    monkeypatch.setattr(
        cli,
        "importlib",
        SimpleNamespace(
            import_module=lambda _n: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )
    with pytest.raises(SystemExit) as ei:
        loader._load_exp_class_instance("x.y", "train", "trainer", "exp")
    assert "Failed to import" in str(ei.value)


def test_experiment_loader_no_suitable_class_lists_available(
    monkeypatch: pytest.MonkeyPatch,
):
    loader = cli.ExperimentLoader()

    class A:  # noqa: D401
        pass

    class B:  # noqa: D401
        pass

    fake_mod = SimpleNamespace(A=A, B=B)
    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: fake_mod)
    )

    with pytest.raises(SystemExit) as ei:
        loader._load_exp_class_instance("mod", "prepare", "preparer", "expZ")
    s = str(ei.value)
    assert "No suitable class" in s and "A" in s and "B" in s


def test_experiment_loader_caches_instances(monkeypatch: pytest.MonkeyPatch):
    loader = cli.ExperimentLoader()

    class Prep:  # noqa: D401
        def prepare(self):  # noqa: D401
            return None

    fake_mod = SimpleNamespace(Prep=Prep)
    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: fake_mod)
    )

    i1 = loader._load_exp_class_instance("m", "prepare", "preparer", "expA")
    i2 = loader._load_exp_class_instance("m", "prepare", "preparer", "expA")
    assert i1 is i2


def test_complete_experiments_handles_iterdir_error(monkeypatch: pytest.MonkeyPatch):
    # Make root exist but iterdir raise
    class FakeRoot:
        def exists(self):  # noqa: D401
            return True

        def iterdir(self):  # noqa: D401
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_experiments_root", lambda: FakeRoot())
    out = cli._complete_experiments(SimpleNamespace(), "b")
    assert out == []


def test_complete_experiments_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Create fake experiments root with mixed contents
    (tmp_path / "bundestag_char").mkdir()
    (tmp_path / "bundestag_char" / "config.toml").write_text("[x]\n")
    (tmp_path / "bundestag_tiktoken").mkdir()
    (tmp_path / "bundestag_tiktoken" / "config.toml").write_text("[x]\n")
    (tmp_path / "foo").mkdir()  # no config.toml -> should be ignored
    (tmp_path / "afile").write_text("not a dir")  # not a directory

    class FakeRoot:
        def __init__(self, p: Path):  # noqa: D401
            self._p = p

        def exists(self):  # noqa: D401
            return True

        def iterdir(self):  # noqa: D401
            return list(self._p.iterdir())

    monkeypatch.setattr(cli, "_experiments_root", lambda: FakeRoot(tmp_path))

    # No prefix -> both valid experiments sorted
    all_names = cli._complete_experiments(SimpleNamespace(), "")
    assert all_names == ["bundestag_char", "bundestag_tiktoken"]

    # Prefix 'b' -> both
    b_names = cli._complete_experiments(SimpleNamespace(), "b")
    assert b_names == ["bundestag_char", "bundestag_tiktoken"]

    # More specific prefix -> single match
    c_names = cli._complete_experiments(SimpleNamespace(), "bundestag_c")
    assert c_names == ["bundestag_char"]


def test_log_command_status_not_set_and_listdir_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO)

    # Not set paths on PreparerConfig
    cli._log_command_status("prepare", PreparerConfig())
    assert any("<not set>" in r.message for r in caplog.records)

    # Existing path but listing fails
    d = tmp_path / "dir"
    d.mkdir()

    # Subclass the concrete system Path type (e.g., PosixPath) for safe override
    class _Path(type(tmp_path)):
        def iterdir(self):  # noqa: D401
            raise RuntimeError("boom")

    # Build a TrainerConfig with runtime.out_dir replaced by our failing path
    base = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=d),
    )
    # Monkeypatch the out_dir attribute via model_copy
    bad_rt = base.runtime.model_copy(update={"out_dir": _Path(str(d))})
    bad = base.model_copy(update={"runtime": bad_rt})

    cli._log_command_status("train", bad)
    # Should log '(exists)' fallback
    assert any("(exists)" in r.message for r in caplog.records)

    # Missing path branch
    missing_dir = tmp_path / "does_not_exist"
    assert not missing_dir.exists()
    missing_cfg = base.model_copy(
        update={"runtime": base.runtime.model_copy(update={"out_dir": missing_dir})}
    )
    cli._log_command_status("train", missing_cfg)
    assert any("(missing)" in r.message for r in caplog.records)


class _Ctx:
    def __init__(self):
        self.obj = {}

    def ensure_object(self, typ):  # noqa: D401
        if not isinstance(self.obj, dict):
            self.obj = {}
        return self.obj


def test_ensure_loaded_cache_key_mismatch_calls_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    ctx = _Ctx()
    ctx.ensure_object(dict)
    # Seed cache with different experiment
    ctx.obj["loaded_cache"] = {
        "key": ("expX", None),
        "cfg_path": tmp_path,
        "app": cli.AppConfig(train=None, sample=None),
        "prep": PreparerConfig(),
    }

    called = {"n": 0}

    def fake_loader(exp: str, exp_config: Path | None):  # noqa: D401
        called["n"] += 1
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "load_app_config", fake_loader)

    _ = cli.ensure_loaded(ctx, "expY")
    assert called["n"] == 1


def test_run_or_exit_custom_exit_code(capsys):
    def boom():  # noqa: D401
        raise RuntimeError("bad")

    with pytest.raises(typer.Exit) as ei:  # type: ignore[name-defined]
        cli.run_or_exit(boom, exception_exit_code=5)
    assert ei.value.exit_code == 5
    assert "bad" in capsys.readouterr().out


def test_run_or_exit_keyboard_interrupt_with_message(capsys):
    def boom():  # noqa: D401
        raise KeyboardInterrupt()

    # Should print provided message and not re-raise as typer.Exit
    cli.run_or_exit(
        boom, keyboard_interrupt_msg="\nInterrupted!", exception_exit_code=9
    )
    out = capsys.readouterr().out
    assert "Interrupted!" in out


def test_run_or_exit_keyboard_interrupt_no_message(capsys):
    def boom():  # noqa: D401
        raise KeyboardInterrupt()

    # No message provided: should not print anything and not raise
    cli.run_or_exit(boom, keyboard_interrupt_msg=None, exception_exit_code=9)
    out = capsys.readouterr().out
    assert out == ""


def test_cmd_train_uses_loaded_error_and_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # Ensure ensure_loaded returns app with train=None
    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)

    # Build ctx with loaded_errors matching the key and custom train error
    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {
                    "key": ("expZ", None),
                    "train": "custom-train-error",
                    "sample": None,
                },
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_train(ctx, "expZ")
    assert ei.value.exit_code == 2
    out = capsys.readouterr().out
    assert "custom-train-error" in out


def test_cmd_sample_uses_loaded_error_and_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # ensure_loaded returns app with sample=None
    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {
                    "key": ("expY", None),
                    "train": None,
                    "sample": "custom-sample-error",
                },
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_sample(ctx, "expY")
    assert ei.value.exit_code == 2
    out = capsys.readouterr().out
    assert "custom-sample-error" in out


def test_cmd_sample_speakger_uses_unified_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Ensure ensure_loaded returns a non-None sample config so cmd_sample proceeds
    expected_cfg = tmp_path / "cfg.toml"

    from types import SimpleNamespace

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        # Return a lightweight object with the required attributes to avoid Pydantic validation
        return (
            expected_cfg,
            SimpleNamespace(train=None, sample=SimpleNamespace()),
            PreparerConfig(),
        )

    called = {"n": 0, "args": None}

    def fake_run_sample(experiment, sample_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["args"] = (experiment, sample_cfg, cfg_path)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_sample", fake_run_sample)

    ctx = Ctx()
    cli.cmd_sample(ctx, "speakger")
    assert (
        called["n"] == 1
        and called["args"][0] == "speakger"
        and called["args"][2] == expected_cfg
    )


def test_run_analyze_unsupported_experiment():
    with pytest.raises(RuntimeError) as ei:
        cli._run_analyze("other_exp", host="127.0.0.1", port=9999, open_browser=False)
    assert "only 'bundestag_char'" in str(ei.value)


def test_cmd_convert_only_bundestag_char(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # Non-supported experiment prints and exits code 2
    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_convert(ctx, "not_supported")
    assert ei.value.exit_code == 2
    assert "supports only 'bundestag_char'" in capsys.readouterr().out


def test_cmd_convert_success_and_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # ensure_loaded should be called but we only care it returns (cfg_path, _, _)
    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)

    called = {"ok": 0, "args": None}

    class _ExportCfg:
        def __init__(
            self,
            enabled,
            export_dir,
            model_name,
            quant,
            template=None,
            convert_bin=None,
            quant_bin=None,
        ):  # noqa: D401
            self.enabled = enabled
            self.export_dir = export_dir
            self.model_name = model_name
            self.quant = quant
            self.template = template
            self.convert_bin = convert_bin
            self.quant_bin = quant_bin

    class OkMod:
        OllamaExportConfig = _ExportCfg

        def convert(self, export_cfg, out_dir, read_policy):  # noqa: D401
            called["ok"] += 1
            called["args"] = (export_cfg, out_dir, read_policy)

    # Success path
    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: OkMod())
    )

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_convert(ctx, "bundestag_char")
    assert called["ok"] == 1 and isinstance(called["args"][1], Path)

    # SystemExit inside converter is mapped to typer.Exit with same code and message echoed
    class ExitMod:
        OllamaExportConfig = _ExportCfg

        def convert(self, export_cfg, out_dir, read_policy):  # noqa: D401
            raise SystemExit("inner-msg")

    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: ExitMod())
    )
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_convert(ctx, "bundestag_char")
    # Current implementation forwards SystemExit.code even if it is a string
    assert ei.value.exit_code == "inner-msg"
    assert "inner-msg" in capsys.readouterr().out

    # Generic exception maps to exit code 1 and echoes message
    class ErrMod:
        OllamaExportConfig = _ExportCfg

        def convert(self, export_cfg, out_dir, read_policy):  # noqa: D401
            raise RuntimeError("boom")

    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: ErrMod())
    )
    with pytest.raises(typer.Exit) as ei2:
        cli.cmd_convert(ctx, "bundestag_char")
    assert ei2.value.exit_code == 1
    assert "boom" in capsys.readouterr().out


def test_cmd_prepare_happy_calls_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # ensure_loaded returns preparer config in position 3
    from ml_playground.prepare import PreparerConfig

    pcfg = PreparerConfig()

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return tmp_path / "cfg.toml", cli.AppConfig(train=None, sample=None), pcfg

    called = {"n": 0, "arg": None}

    def fake_run_prepare(experiment, prep_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["arg"] = prep_cfg

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_prepare", fake_run_prepare)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_prepare(ctx, "expX")
    assert called["n"] == 1 and called["arg"] is pcfg


def test_cmd_train_happy_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # ensure_loaded returns app.train config
    from ml_playground.config import (
        TrainerConfig,
        ModelConfig,
        DataConfig,
        OptimConfig,
        LRSchedule,
        RuntimeConfig,
    )

    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return tmp_path / "cfg.toml", cli.AppConfig(train=tcfg, sample=None), None

    called = {"n": 0, "arg": None}

    def fake_run_train(experiment, train_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["arg"] = train_cfg

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_train", fake_run_train)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {"key": ("expA", None), "train": None, "sample": None},
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_train(ctx, "expA")
    assert called["n"] == 1 and called["arg"] is tcfg


def test_cmd_sample_happy_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from ml_playground.config import SamplerConfig, RuntimeConfig, SampleConfig

    scfg = SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=SampleConfig())

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return tmp_path / "cfg.toml", cli.AppConfig(train=None, sample=scfg), None

    called = {"n": 0, "arg": None}

    def fake_run_sample(experiment, sample_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["arg"] = sample_cfg

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_sample", fake_run_sample)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {"key": ("expB", None), "train": None, "sample": None},
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_sample(ctx, "expB")
    assert called["n"] == 1 and called["arg"] is scfg


def test__load_sample_config_missing_sample_block(tmp_path: Path):
    # Config with no [sample] block at all
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[train]\n")

    with pytest.raises(ValueError, match=r"must contain a \[sample\] section"):
        config_loader.load_sample_config(cfg_path)


def test__load_sample_config_unknown_top_key(tmp_path: Path):
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("""
    [sample]
    bad = 1
    [sample.sample]
    """)
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "Unknown key(s) in [sample]" in str(ei.value)


def test__load_sample_config_missing_sample_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Disable defaults so missing [sample.sample] is not filled from defaults
    orig_load = cli.tomllib.load

    def _no_defaults_load(f):  # noqa: D401
        name = getattr(f, "name", "")
        if isinstance(name, str) and name.endswith("default_config.toml"):
            return {}
        return orig_load(f)

    monkeypatch.setattr(cli.tomllib, "load", _no_defaults_load)
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("""
    [sample]
    runtime_ref = "train.runtime"
    """)
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "Missing required section [sample]" in str(ei.value)


def test__load_sample_config_requires_runtime_or_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Disable defaults so runtime/runtime_ref isn't provided by defaults
    orig_load = cli.tomllib.load

    def _no_defaults_load(f):  # noqa: D401
        name = getattr(f, "name", "")
        if isinstance(name, str) and name.endswith("default_config.toml"):
            return {}
        return orig_load(f)

    monkeypatch.setattr(cli.tomllib, "load", _no_defaults_load)
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("""
    [sample]
    [sample.sample]
    """)
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "requires either [sample.runtime] or sample.runtime_ref" in str(ei.value)


def test__load_sample_config_runtime_ref_unsupported_value(tmp_path: Path):
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("""
    [sample]
    runtime_ref = "other.ref"
    [sample.sample]
    """)
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "Unsupported sample.runtime_ref" in str(ei.value)


def test__load_sample_config_runtime_ref_missing_train_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # runtime_ref points to train.runtime, but no train.runtime provided in exp nor defaults
    orig_load = cli.tomllib.load

    def _no_defaults_load(f):  # noqa: D401
        name = getattr(f, "name", "")
        if isinstance(name, str) and name.endswith("default_config.toml"):
            return {}
        return orig_load(f)

    monkeypatch.setattr(cli.tomllib, "load", _no_defaults_load)
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("""
    [sample]
    runtime_ref = "train.runtime"
    [sample.sample]
    """)
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "points to 'train.runtime'" in str(ei.value)


def test_cmd_loop_happy_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from ml_playground.config import (
        TrainerConfig,
        ModelConfig,
        DataConfig,
        OptimConfig,
        LRSchedule,
        RuntimeConfig,
        SamplerConfig,
        SampleConfig,
    )
    from ml_playground.prepare import PreparerConfig

    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    scfg = SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=SampleConfig())
    pcfg = PreparerConfig()

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return tmp_path / "cfg.toml", cli.AppConfig(train=tcfg, sample=scfg), pcfg

    called = {"n": 0, "args": None}

    def fake_run_loop(experiment, prep_cfg, train_cfg, sample_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["args"] = (experiment, prep_cfg, train_cfg, sample_cfg, cfg_path)

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_loop", fake_run_loop)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {
                    "key": ("expLoop", None),
                    "train": None,
                    "sample": None,
                },
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_loop(ctx, "expLoop")
    assert called["n"] == 1

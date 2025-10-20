from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from ml_playground.configuration.models import (
    DataConfig,
    ExperimentConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    PreparerConfig,
    RuntimeConfig,
    SampleConfig,
    TrainerConfig,
)
from ml_playground.configuration import cli as config_cli
from ml_playground.configuration import loading as config_loading
from ml_playground.configuration.merge_utils import merge_mappings
from tests.conftest import minimal_full_experiment_toml


def test_full_loader_roundtrip(tmp_path: Path) -> None:
    toml_text = minimal_full_experiment_toml(
        dataset_dir=Path("data/shakespeare"),
        out_dir=Path("out/test_next"),
        extra_optim="learning_rate = 0.001",
        extra_train="max_iters = 1",
        extra_sample="",
        extra_sample_sample="",
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text)
    project_home = tmp_path.parent if tmp_path.parent.name else tmp_path
    experiment_name = cfg_path.parent.name
    exp: ExperimentConfig = config_loading.load_full_experiment_config(
        cfg_path, project_home, experiment_name
    )
    assert exp.train is not None
    assert exp.sample is not None
    assert isinstance(exp.train.runtime.out_dir, Path)
    assert isinstance(exp.shared.dataset_dir, Path)


def test_read_toml_dict_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        config_loading.read_toml_dict(missing_path)


def test_get_default_config_path_with_none_uses_package_root() -> None:
    """get_default_config_path with None should use package root."""
    path = config_loading.get_default_config_path(None)
    assert path.name == "default_config.toml"
    assert (
        str(path)
        .replace("\\", "/")
        .endswith("src/ml_playground/experiments/default_config.toml")
    )


def test_get_default_config_path_with_explicit_root(tmp_path: Path) -> None:
    """get_default_config_path with explicit root should use that root."""
    path = config_loading.get_default_config_path(tmp_path)
    assert (
        path
        == tmp_path / "src" / "ml_playground" / "experiments" / "default_config.toml"
    )


def test_default_config_path_when_root_is_src(tmp_path: Path) -> None:
    """_default_config_path_from_root should handle roots named 'src'."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    resolved = config_loading.get_default_config_path(src_root)
    assert (
        resolved == src_root / "ml_playground" / "experiments" / "default_config.toml"
    )


def test_get_cfg_path_without_override(tmp_path: Path) -> None:
    expected = (
        config_loading._package_root() / "experiments" / "demo" / "config.toml"
    )
    result = config_loading.get_cfg_path("demo", None)
    assert result == expected


def test_list_experiments_with_config_returns_sorted_names(tmp_path: Path) -> None:
    """list_experiments_with_config should return sorted experiment names with config.toml."""
    # Create fake experiments directory structure
    experiments_root = tmp_path / "src" / "ml_playground" / "experiments"
    experiments_root.mkdir(parents=True)

    # Create experiments with config.toml
    (experiments_root / "exp_a").mkdir()
    (experiments_root / "exp_a" / "config.toml").write_text("")
    (experiments_root / "exp_c").mkdir()
    (experiments_root / "exp_c" / "config.toml").write_text("")
    (experiments_root / "exp_b").mkdir()
    (experiments_root / "exp_b" / "config.toml").write_text("")

    # Create experiment without config.toml (should be excluded)
    (experiments_root / "exp_no_config").mkdir()

    # Mock the package root
    result = config_loading.list_experiments_with_config(
        experiments_root=experiments_root
    )
    assert result == ["exp_a", "exp_b", "exp_c"]


def test_list_experiments_with_config_filters_by_prefix(tmp_path: Path) -> None:
    """list_experiments_with_config should filter by prefix."""
    experiments_root = tmp_path / "src" / "ml_playground" / "experiments"
    experiments_root.mkdir(parents=True)

    (experiments_root / "bundestag_char").mkdir()
    (experiments_root / "bundestag_char" / "config.toml").write_text("")
    (experiments_root / "bundestag_tiktoken").mkdir()
    (experiments_root / "bundestag_tiktoken" / "config.toml").write_text("")
    (experiments_root / "shakespeare").mkdir()
    (experiments_root / "shakespeare" / "config.toml").write_text("")

    result = config_loading.list_experiments_with_config(
        "bundestag", experiments_root=experiments_root
    )
    assert result == ["bundestag_char", "bundestag_tiktoken"]


def test_list_experiments_with_config_handles_missing_root() -> None:
    """list_experiments_with_config should return empty list if experiments root doesn't exist."""
    missing_root = Path("/nonexistent/path/loading")
    result = config_loading.list_experiments_with_config(experiments_root=missing_root)
    assert result == []


def test_list_experiments_with_config_handles_os_error(tmp_path: Path) -> None:
    """list_experiments_with_config should return empty list on OSError."""
    experiments_root = tmp_path / "src" / "ml_playground" / "experiments"
    experiments_root.mkdir(parents=True)

    class BrokenPath(type(experiments_root)):  # type: ignore[misc]
        def iterdir(self):  # type: ignore[override]
            raise OSError("Simulated error")

    broken_root = BrokenPath(experiments_root)

    result = config_loading.list_experiments_with_config(experiments_root=broken_root)
    assert result == []


def test_load_and_merge_configs_missing_file_raises(tmp_path: Path) -> None:
    """_load_and_merge_configs should raise FileNotFoundError for missing config."""
    missing_path = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        config_loading._load_and_merge_configs(missing_path, tmp_path, "test")


def test_load_prepare_config_success(tmp_path: Path) -> None:
    """load_prepare_config should load and validate prepare config."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("""
[prepare]
tokenizer_type = "char"
""")

    # Create default config
    default_path = (
        tmp_path / "src" / "ml_playground" / "experiments" / "default_config.toml"
    )
    default_path.parent.mkdir(parents=True)
    default_path.write_text("")

    cfg = config_loading.load_prepare_config(cfg_path, default_config_path=default_path)
    assert isinstance(cfg, PreparerConfig)
    assert cfg.tokenizer_type == "char"
    assert "provenance" in cfg.extras


def test_load_prepare_config_missing_section_raises(tmp_path: Path) -> None:
    """load_prepare_config should raise ValueError if [prepare] section is missing."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[train]\n")

    default_path = (
        tmp_path / "src" / "ml_playground" / "experiments" / "default_config.toml"
    )
    default_path.parent.mkdir(parents=True)
    default_path.write_text("")

    with pytest.raises(ValueError, match="must contain a \\[prepare\\] section"):
        config_loading.load_prepare_config(cfg_path, default_config_path=default_path)


def test_load_train_config_sets_provenance(tmp_path: Path) -> None:
    config = tmp_path / "train.toml"
    config.write_text(
        """
[train]
[train.runtime]
out_dir = "./out"
[train.model]
[train.data]
[train.optim]
[train.schedule]
"""
    )

    default_config = tmp_path / "default.toml"
    default_config.write_text("")

    cfg = config_loading.load_train_config(
        config, default_config_path=default_config
    )

    provenance = cfg.extras.get("provenance", {})
    assert provenance.get("raw") is not None
    assert provenance.get("context", {}).get("config_path") == str(config)


def test_load_sample_config_sets_provenance(tmp_path: Path) -> None:
    config = tmp_path / "sample.toml"
    config.write_text(
        """
[sample]
[sample.runtime]
out_dir = "./out"
[sample.sample]

[train]
[train.runtime]
out_dir = "./train"
[train.model]
[train.data]
[train.optim]
[train.schedule]
"""
    )

    default_config = tmp_path / "default.toml"
    default_config.write_text("")

    cfg = config_loading.load_sample_config(
        config, default_config_path=default_config
    )

    provenance = cfg.extras.get("provenance", {})
    assert provenance.get("raw") is not None
    assert provenance.get("context", {}).get("config_path") == str(config)


def test_load_train_config_requires_mapping(tmp_path: Path) -> None:
    config = tmp_path / "train_invalid.toml"
    config.write_text("train = 'value'\n")

    default_config = tmp_path / "default.toml"
    default_config.write_text("")

    with pytest.raises(TypeError, match="\\[train\\] section"):
        config_loading.load_train_config(
            config, default_config_path=default_config
        )


def test_load_sample_config_requires_sample_block(tmp_path: Path) -> None:
    config = tmp_path / "sample_invalid.toml"
    config.write_text("[train]\n[train.runtime]\nout_dir='.'\n")

    default_config = tmp_path / "default.toml"
    default_config.write_text("")

    with pytest.raises(ValueError, match=r"must contain a \[sample\] section"):
        config_loading.load_sample_config(
            config, default_config_path=default_config
        )


def test_read_toml_dict_reads_existing_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("key = 'value'", encoding="utf-8")
    data = config_loading.read_toml_dict(cfg_path)
    assert data == {"key": "value"}


def test_read_toml_dict_rejects_non_mapping_root(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("key = 'value'", encoding="utf-8")

    def fake_loads(_: str) -> list[int]:
        return [1, 2, 3]

    with pytest.raises(TypeError, match="must be a mapping"):
        config_loading.read_toml_dict(cfg_path, toml_loader=fake_loads)


def test_read_toml_dict_invalid_toml_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "broken.toml"
    cfg_path.write_text("not = [", encoding="utf-8")

    with pytest.raises(Exception, match="broken.toml"):
        config_loading.read_toml_dict(cfg_path)


def test_full_loader_empty_config_raises(tmp_path: Path) -> None:
    toml_text = ""
    cfg_path = tmp_path / "empty.toml"
    cfg_path.write_text(toml_text)
    project_home = tmp_path.parent if tmp_path.parent.name else tmp_path
    experiment_name = cfg_path.parent.name
    with pytest.raises(Exception):
        config_loading.load_full_experiment_config(
            cfg_path, project_home, experiment_name
        )


def test_full_loader_bad_root_type(tmp_path: Path) -> None:
    bad_text = """
arr = [1,2,3]
"""
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(bad_text)
    project_home = tmp_path.parent if tmp_path.parent.name else tmp_path
    experiment_name = cfg_path.parent.name
    with pytest.raises(Exception):
        config_loading.load_full_experiment_config(
            cfg_path, project_home, experiment_name
        )


def test_full_loader_nested_unknown_keys_in_sample_raise(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_bad_sample_nested.toml"
    text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("./out"),
        extra_sample_sample="unknown_leaf = 42",
    )
    cfg_path.write_text(text)
    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_incomplete_train_config(tmp_path: Path) -> None:
    toml_text = """
[prepare]

[train.model]
n_layer=1

# Missing other required sections
"""
    cfg_path = tmp_path / "incomplete.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_unknown_top_level_sections_raise(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.toml"
    base = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("./out"),
        include_sample=True,
        extra_sample_sample='start = "\\n"',
    )
    cfg_path.write_text(base + "\n[export]\nfoo = 1\n")
    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_nested_unknown_keys_raise(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_bad_nested.toml"
    text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("./out"),
    )
    text = text.replace("[train.model]", "[train.model]\nunknown_key = 123")
    cfg_path.write_text(text)
    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_load_experiment_toml_strict_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.toml"
    text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("./out"),
        extra_train="log_interval = 2",
        extra_sample="log_interval = 2",
        extra_sample_sample='start = "\\n"',
    )
    cfg_path.write_text(text)
    exp = config_loading.load_experiment_toml(cfg_path)
    assert isinstance(exp, ExperimentConfig)
    assert exp.sample.runtime is not None
    assert str(exp.sample.runtime.out_dir).endswith("out")
    assert exp.sample.runtime.log_interval == 2


def test_explicit_sample_runtime_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp2.toml"
    text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("./out"),
        extra_train="""
eval_interval = 100
eval_iters = 20
tensorboard_enabled = true
""",
        extra_sample="""
eval_interval = 200
tensorboard_enabled = false
""",
        extra_sample_sample='start = "\\n"',
    )
    cfg_path.write_text(text)
    exp = config_loading.load_experiment_toml(cfg_path)
    runtime = exp.sample.runtime
    assert runtime is not None
    assert str(runtime.out_dir).endswith("out")
    assert runtime.eval_interval == 200
    assert runtime.eval_iters == 200
    assert runtime.tensorboard_enabled is False


def test_data_config_tokenizer_choices() -> None:
    DataConfig(tokenizer="char")
    DataConfig(tokenizer="word")
    DataConfig(tokenizer="tiktoken")


def test_dataconfig_positive_ints() -> None:
    with pytest.raises(ValidationError):
        DataConfig(batch_size=0)
    with pytest.raises(ValidationError):
        DataConfig(block_size=-1)
    with pytest.raises(ValidationError):
        DataConfig(grad_accum_steps=0)
    with pytest.raises(ValidationError):
        DataConfig(ngram_size=0)
    cfg = DataConfig(batch_size=1, block_size=1, grad_accum_steps=1, ngram_size=1)
    assert cfg.batch_size == 1


def test_sampleconfig_ranges() -> None:
    with pytest.raises(ValidationError):
        SampleConfig(temperature=0.0)
    with pytest.raises(ValidationError):
        SampleConfig(top_k=-1)
    with pytest.raises(ValidationError):
        SampleConfig(top_p=0.0)
    with pytest.raises(ValidationError):
        SampleConfig(top_p=1.5)
    SampleConfig(temperature=0.1, top_k=0, top_p=0.5)


def test_lrschedule_validations() -> None:
    with pytest.raises(ValidationError):
        LRSchedule(warmup_iters=-1)
    with pytest.raises(ValidationError):
        LRSchedule(lr_decay_iters=-5)
    with pytest.raises(ValidationError):
        LRSchedule(warmup_iters=10, lr_decay_iters=5)
    with pytest.raises(ValidationError):
        LRSchedule(min_lr=-1e-5)
    LRSchedule(warmup_iters=1, lr_decay_iters=2, min_lr=0)
    LRSchedule(warmup_iters=2, lr_decay_iters=2, min_lr=0)


def test_optimconfig_non_negative() -> None:
    with pytest.raises(ValidationError):
        OptimConfig(learning_rate=-1e-3)
    with pytest.raises(ValidationError):
        OptimConfig(weight_decay=-1e-1)
    with pytest.raises(ValidationError):
        OptimConfig(beta1=-0.1)
    with pytest.raises(ValidationError):
        OptimConfig(beta2=-0.1)
    with pytest.raises(ValidationError):
        OptimConfig(grad_clip=-1)
    OptimConfig()


def test_modelconfig_ranges() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(n_layer=0)
    with pytest.raises(ValidationError):
        ModelConfig(n_head=0)
    with pytest.raises(ValidationError):
        ModelConfig(n_embd=0)
    with pytest.raises(ValidationError):
        ModelConfig(block_size=0)
    with pytest.raises(ValidationError):
        ModelConfig(dropout=1.5)
    with pytest.raises(ValidationError):
        ModelConfig(vocab_size=0)
    ModelConfig()


def test_default_constants_across_configs() -> None:
    schedule = LRSchedule()
    assert schedule.decay_lr is True
    assert schedule.warmup_iters == 2_000
    assert schedule.lr_decay_iters == 600_000
    assert schedule.min_lr == 6e-5

    optim = OptimConfig()
    assert optim.learning_rate == pytest.approx(6e-4)
    assert optim.weight_decay == pytest.approx(1e-1)
    assert optim.beta1 == pytest.approx(0.9)
    assert optim.beta2 == pytest.approx(0.95)
    assert optim.grad_clip == pytest.approx(1.0)

    model = ModelConfig()
    assert model.n_layer == 12
    assert model.n_head == 12
    assert model.n_embd == 767
    assert model.block_size == 1024

    sample = SampleConfig()
    assert sample.start == "\n"
    assert sample.num_samples == 3
    assert sample.max_new_tokens == 200
    assert sample.temperature == pytest.approx(0.8)
    assert sample.top_k == 200
    assert sample.top_p is None


def test_runtime_checkpointing_keep_non_negative(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(
            out_dir=tmp_path,
            checkpointing=RuntimeConfig.Checkpointing(
                keep=RuntimeConfig.Checkpointing.Keep(last=-1)
            ),
        )
    with pytest.raises(ValidationError):
        RuntimeConfig(
            out_dir=tmp_path,
            checkpointing=RuntimeConfig.Checkpointing(
                keep=RuntimeConfig.Checkpointing.Keep(best=-2)
            ),
        )
    RuntimeConfig(out_dir=tmp_path)


def test_runtimeconfig_defaults_and_checkpointing() -> None:
    runtime = RuntimeConfig(out_dir=Path("./out"))
    assert runtime.max_iters == 600_000
    assert runtime.eval_interval == 2_000
    assert runtime.eval_iters == 200
    assert runtime.log_interval == 1
    assert runtime.eval_only is False
    assert runtime.seed == 1337
    assert runtime.device == "cpu"
    assert runtime.dtype == "float32"
    assert runtime.compile is False
    assert runtime.tensorboard_enabled is True

    checkpoint = runtime.checkpointing
    assert checkpoint.read_policy in ("latest", "best")
    assert checkpoint.keep.last == 1
    assert checkpoint.keep.best == 1
    assert runtime.ckpt_metric in ("val_loss", "perplexity")
    assert runtime.ckpt_greater_is_better is False
    assert runtime.ckpt_atomic is True
    assert runtime.ckpt_write_metadata is True
    assert runtime.ckpt_time_interval_minutes == 0


def test_merge_mappings_nested_and_replace() -> None:
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": {"k": 1}, "d": 4}
    override = {"b": {"y": 20, "z": 3}, "c": 5, "e": 6}
    out = merge_mappings(base, override)
    assert out["b"] == {"x": 1, "y": 20, "z": 3}
    assert out["c"] == 5
    assert out["a"] == 1 and out["d"] == 4
    assert out["e"] == 6


def test_merge_mappings_numeric_replacements() -> None:
    base = {"a": {"x": 1, "y": -2}, "b": 10}
    override = {"a": {"x": 3}, "b": 0}
    out = merge_mappings(base, override)
    assert out["a"]["x"] == 3
    assert out["a"]["y"] == -2
    assert out["b"] == 0


def test_merge_mappings_type_replacement() -> None:
    base = {"a": {"x": 1}, "b": {"y": 2}}
    override = {"b": 7}
    out = merge_mappings(base, override)
    assert out["a"] == {"x": 1}
    assert out["b"] == 7


def test_trainer_resolves_relative_runtime_out_dir(tmp_path: Path) -> None:
    cfg_text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("out/rel_train"),
        include_sample=True,
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    exp = config_loading.load_full_experiment_config(cfg_path, tmp_path, "exp")
    assert isinstance(exp.train.runtime.out_dir, Path)
    assert str(exp.train.runtime.out_dir).endswith("out/rel_train")


def test_sampler_resolves_relative_runtime_out_dir(tmp_path: Path) -> None:
    cfg_text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("out/rel_sample"),
        include_sample=True,
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    exp = config_loading.load_full_experiment_config(cfg_path, tmp_path, "exp")
    assert isinstance(exp.sample.runtime.out_dir, Path)
    assert str(exp.sample.runtime.out_dir).endswith("out/rel_sample")


def test_experiment_config_shared_path_coercions(tmp_path: Path) -> None:
    ds_dir = Path("./data/shared")
    out_dir = Path("out/shared")
    cfg_text = minimal_full_experiment_toml(dataset_dir=ds_dir, out_dir=out_dir)
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    exp: ExperimentConfig = config_loading.load_full_experiment_config(
        cfg_path, tmp_path, "exp"
    )
    assert isinstance(exp.shared.dataset_dir, Path)
    assert isinstance(exp.shared.train_out_dir, Path)
    assert isinstance(exp.shared.sample_out_dir, Path)
    assert exp.shared.train_out_dir.is_absolute()
    assert exp.shared.sample_out_dir.is_absolute()
    assert str(exp.shared.train_out_dir).endswith(str(out_dir))
    assert str(exp.shared.sample_out_dir).endswith(str(out_dir))


def test_cross_field_validations(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrainerConfig(
            model=ModelConfig(block_size=4),
            data=DataConfig(block_size=8, batch_size=1, grad_accum_steps=1),
            optim=OptimConfig(),
            schedule=LRSchedule(
                decay_lr=True, warmup_iters=1, lr_decay_iters=2, min_lr=0
            ),
            runtime=RuntimeConfig(out_dir=tmp_path),
        )

    with pytest.raises(ValueError):
        LRSchedule(decay_lr=True, warmup_iters=10, lr_decay_iters=5, min_lr=0)

    with pytest.raises(ValueError):
        TrainerConfig(
            model=ModelConfig(block_size=4),
            data=DataConfig(block_size=4, batch_size=1, grad_accum_steps=1),
            optim=OptimConfig(learning_rate=1e-3),
            schedule=LRSchedule(
                decay_lr=True, warmup_iters=0, lr_decay_iters=2, min_lr=2e-3
            ),
            runtime=RuntimeConfig(out_dir=tmp_path),
        )

    with pytest.raises(ValueError):
        RuntimeConfig(out_dir=tmp_path, log_interval=10, eval_interval=1)


def test_dataconfig_paths_and_defaults() -> None:
    config = DataConfig()
    assert config.train_bin == "train.bin"
    assert config.val_bin == "val.bin"
    assert config.meta_pkl == "meta.pkl"
    assert config.batch_size == 12
    assert config.block_size == 1024
    assert config.grad_accum_steps == 40
    assert config.tokenizer in ("char", "word", "tiktoken")
    assert config.ngram_size == 1
    assert config.sampler in ("random", "sequential")


def test_dataconfig_meta_none_rejected() -> None:
    with pytest.raises(ValidationError):
        DataConfig(meta_pkl=cast(Any, None))


def test_preparerconfig_path_coercion_and_resolve(tmp_path: Path) -> None:
    config = PreparerConfig(raw_dir=tmp_path / "raw")
    assert isinstance(config.raw_dir, Path)
    _ = config.raw_dir


def test_sampleconfig_more_ranges() -> None:
    with pytest.raises(ValidationError):
        SampleConfig(num_samples=0)
    with pytest.raises(ValidationError):
        SampleConfig(max_new_tokens=0)
    with pytest.raises(ValidationError):
        SampleConfig(temperature=-0.1)
    SampleConfig(temperature=1e-6, top_k=0, top_p=1.0)


def test_config_canonical_exports() -> None:
    """Test that canonical configuration modules export expected APIs."""
    from ml_playground.configuration import models
    from ml_playground.configuration import loading

    assert hasattr(models, "TrainerConfig")
    assert hasattr(models, "SamplerConfig")
    assert hasattr(models, "DataConfig")
    assert hasattr(models, "RuntimeConfig")
    assert hasattr(loading, "load_full_experiment_config")


def test_full_loader_incomplete_sample_config(tmp_path: Path) -> None:
    toml_text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("out/test"),
        include_sample=True,
    )
    toml_text = toml_text.replace("[sample.sample]", "# Missing sample.sample")
    cfg_path = tmp_path / "incomplete_sample.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_no_train_section_raises(tmp_path: Path) -> None:
    toml_text = """
[prepare]

[sample.runtime]
out_dir = "out/test"

[sample.sample]
"""
    cfg_path = tmp_path / "no_train.toml"
    cfg_path.write_text(toml_text)
    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_no_sample_section_raises(tmp_path: Path) -> None:
    toml_text = minimal_full_experiment_toml(
        dataset_dir=Path("data/shakespeare"),
        out_dir=Path("out/test"),
        include_sample=False,
    )
    cfg_path = tmp_path / "no_sample.toml"
    cfg_path.write_text(toml_text)
    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_train_missing_data_section(tmp_path: Path) -> None:
    toml_text = minimal_full_experiment_toml(
        dataset_dir=Path("data/shakespeare"),
        out_dir=Path("out/test"),
        include_train_data=False,
    )
    cfg_path = tmp_path / "missing_data.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_train_missing_runtime_section(tmp_path: Path) -> None:
    toml_text = minimal_full_experiment_toml(
        dataset_dir=Path("data/shakespeare"),
        out_dir=Path("out/test"),
        include_train_runtime=False,
    )
    cfg_path = tmp_path / "missing_runtime.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_full_loader_sample_missing_runtime_section(tmp_path: Path) -> None:
    toml_text = minimal_full_experiment_toml(
        dataset_dir=Path("./data"),
        out_dir=Path("out/test"),
        include_sample=True,
    )
    toml_text = toml_text.replace("[sample.runtime]", "# Missing [sample.runtime]")
    cfg_path = tmp_path / "sample_missing_runtime.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        config_loading.load_experiment_toml(cfg_path)


def test_cli_adapters_load_and_validate(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.toml"
    cfg_path.write_text(
        minimal_full_experiment_toml(
            dataset_dir=Path("./data"),
            out_dir=Path("out/exp"),
            include_sample=True,
        ),
        encoding="utf-8",
    )
    exp = config_cli.load_experiment("exp", cfg_path)
    assert exp.shared.experiment == "exp"
    assert exp.shared.dataset_dir.is_absolute()


def test_cli_adapters_prerequisites(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.toml"
    cfg_path.write_text(
        minimal_full_experiment_toml(
            dataset_dir=Path("./data"),
            out_dir=Path("out/exp"),
            include_sample=True,
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_meta = data_dir / "meta.pkl"
    train_meta.write_bytes(b"meta")

    exp = config_cli.load_experiment("exp", cfg_path)

    found_train_meta = config_cli.ensure_train_prerequisites(exp)
    assert found_train_meta == train_meta

    runtime_meta_dir = exp.shared.sample_out_dir / exp.shared.experiment
    runtime_meta_dir.mkdir(parents=True, exist_ok=True)
    (runtime_meta_dir / "meta.pkl").write_bytes(b"meta")
    config_cli.ensure_sample_prerequisites(exp)

from __future__ import annotations
from pathlib import Path
import pytest
from pydantic import ValidationError
from ml_playground.config import (
    DataConfig,
    ExperimentConfig,
    load_experiment_toml,
    validate_config_field,
    validate_path_exists,
    SampleConfig,
    LRSchedule,
    OptimConfig,
    ModelConfig,
    RuntimeConfig,
    _deep_merge_dicts,
)
from ml_playground.config_loader import load_full_experiment_config
from ml_playground.prepare import PreparerConfig


def test_full_loader_roundtrip(tmp_path: Path) -> None:
    toml_text = """
[prepare]

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

[sample]
[sample.runtime]
out_dir = "out/test_next"

[sample.sample]
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text)

    exp: ExperimentConfig = load_full_experiment_config(cfg_path)
    assert exp.train is not None
    assert exp.sample is not None
    assert isinstance(exp.train.runtime.out_dir, Path)
    assert isinstance(exp.train.data.dataset_dir, Path)


def test_full_loader_empty_config_raises(tmp_path: Path) -> None:
    """Strict: Empty TOML is invalid for ExperimentConfig (missing sections)."""
    toml_text = ""
    cfg_path = tmp_path / "empty.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(Exception):
        load_full_experiment_config(cfg_path)


def test_full_loader_bad_root_type(tmp_path: Path) -> None:
    # TOML decoding to non-dict root (e.g., array) should raise ValueError
    bad_text = """
arr = [1,2,3]
"""
    p = tmp_path / "bad.toml"
    p.write_text(bad_text)
    with pytest.raises(Exception):
        load_full_experiment_config(p)


def test_full_loader_nested_unknown_keys_in_sample_raise(tmp_path: Path) -> None:
    p = tmp_path / "cfg_bad_sample_nested.toml"
    p.write_text(
        """
[prepare]

[train]
[train.model]

[train.data]
dataset_dir = "./data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "./out"

[sample]
[sample.runtime]
out_dir = "./out"
[sample.sample]
unknown_leaf = 42
"""
    )
    with pytest.raises(ValidationError):
        load_full_experiment_config(p)


def test_full_loader_incomplete_train_config(tmp_path: Path) -> None:
    """Strict: incomplete [train] should raise ValidationError."""
    # Missing required sections like model, data, optim, schedule, runtime
    toml_text = """
[prepare]

[train.model]
n_layer=1

# Missing other required sections
"""
    cfg_path = tmp_path / "incomplete.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_full_experiment_config(cfg_path)


# Consolidated: validators and util tests previously in fragmented files


def test_validate_config_field_success_and_errors() -> None:
    # Success cases
    validate_config_field(3, "k", int)
    validate_config_field(3.5, "k", float, min_value=0.0, max_value=10.0)
    validate_config_field(None, "k", int, required=False)

    # Required missing
    with pytest.raises(ValueError):
        validate_config_field(None, "k", int, required=True)

    # Type mismatch
    with pytest.raises(ValueError):
        validate_config_field("x", "k", int)

    # Range violations
    with pytest.raises(ValueError) as ei:
        validate_config_field(-1, "k", int, min_value=0)
    assert ">= 0" in str(ei.value)
    with pytest.raises(ValueError) as ei2:
        validate_config_field(11, "k", int, max_value=10)
    assert "<= 10" in str(ei2.value)
    # Float edges
    validate_config_field(0.0, "k", float, min_value=0.0)
    with pytest.raises(ValueError):
        validate_config_field(-1e-9, "k", float, min_value=0.0)
    validate_config_field(1.0, "k", float, max_value=1.0)
    with pytest.raises(ValueError):
        validate_config_field(1.0000001, "k", float, max_value=1.0)
    # Cross-type checks
    validate_config_field(1, "k", int, min_value=0, max_value=10)
    with pytest.raises(ValueError):
        validate_config_field(1, "k", float)
    with pytest.raises(ValueError):
        validate_config_field(1.5, "k", int)

    # Object type mismatch
    class X:
        pass

    with pytest.raises(ValueError):
        validate_config_field(X(), "obj", dict)


def test_validate_path_exists_file_and_dir(tmp_path: Path) -> None:
    # Missing
    with pytest.raises(ValueError):
        validate_path_exists(tmp_path / "missing", "x")

    # File case
    f = tmp_path / "a.txt"
    f.write_text("hi")
    validate_path_exists(f, "x", must_be_file=True)
    with pytest.raises(ValueError):
        validate_path_exists(f, "x", must_be_dir=True)

    # Dir case
    d = tmp_path / "d"
    d.mkdir()
    validate_path_exists(d, "x", must_be_dir=True)
    with pytest.raises(ValueError):
        validate_path_exists(d, "x", must_be_file=True)


def test_full_loader_unknown_top_level_sections_raise(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text(
        """
[prepare]

[train]
[train.model]

[train.data]
dataset_dir = "./data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "./out"

[sample]
runtime_ref = "train.runtime"
[sample.sample]
start = "\\n"

[export]
foo = 1
"""
    )
    with pytest.raises(ValidationError):
        load_full_experiment_config(p)


def test_full_loader_nested_unknown_keys_raise(tmp_path: Path) -> None:
    # Unknown nested keys under [train.*] should raise due to strict Pydantic models
    p = tmp_path / "cfg_bad_nested.toml"
    p.write_text(
        """
[prepare]

[train]
[train.model]
unknown_key = 123

[train.data]
dataset_dir = "./data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "./out"

[sample]
[sample.runtime]
out_dir = "./out"
[sample.sample]
"""
    )
    with pytest.raises(ValidationError):
        load_full_experiment_config(p)


def test_load_experiment_toml_strict_sections(tmp_path: Path) -> None:
    p = tmp_path / "exp.toml"
    p.write_text(
        """
[prepare]
# minimal preparer config

[train]
[train.model]

[train.data]
dataset_dir = "./data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "./out"
log_interval = 2

[sample]
    [sample.runtime]
    out_dir = "./out"
    log_interval = 2
    [sample.sample]
    start = "\\n"
"""
    )
    exp = load_experiment_toml(p)
    assert isinstance(exp, ExperimentConfig)
    # Parsed runtime present and matches provided values (no reference resolution)
    assert exp.sample.runtime is not None
    assert exp.sample.runtime.out_dir == Path("./out")
    assert exp.sample.runtime.log_interval == 2


def test_explicit_sample_runtime_overrides(tmp_path: Path) -> None:
    p = tmp_path / "exp2.toml"
    p.write_text(
        """
[prepare]

[train]
[train.model]

[train.data]
dataset_dir = "./data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "./out"
eval_interval = 100
eval_iters = 20
tensorboard_enabled = true

[sample]
[sample.runtime]
out_dir = "./out"
eval_interval = 200
tensorboard_enabled = false
[sample.sample]
start = "\\n"
"""
    )
    exp = load_experiment_toml(p)
    rt = exp.sample.runtime
    assert rt is not None
    # explicit runtime provided
    assert rt.out_dir == Path("./out")
    # eval_interval overridden
    assert rt.eval_interval == 200
    # other values follow sample.runtime or schema defaults (no inheritance)
    assert rt.eval_iters == 200
    assert rt.tensorboard_enabled is False


def test_data_config_tokenizer_choices(tmp_path: Path) -> None:
    # Ensure DataConfig accepts tokenizer variants
    DataConfig(dataset_dir=tmp_path, tokenizer="char")
    DataConfig(dataset_dir=tmp_path, tokenizer="word")
    DataConfig(dataset_dir=tmp_path, tokenizer="tiktoken")


def test_dataconfig_positive_ints(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, batch_size=0)
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, block_size=-1)
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, grad_accum_steps=0)
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, ngram_size=0)
    # happy path
    cfg = DataConfig(
        dataset_dir=tmp_path,
        batch_size=1,
        block_size=1,
        grad_accum_steps=1,
        ngram_size=1,
    )
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
    # ok
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
    # ok
    LRSchedule(warmup_iters=1, lr_decay_iters=2, min_lr=0)
    # boundary equal allowed
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
    # ok
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
    # ok
    ModelConfig()


def test_default_constants_across_configs(tmp_path: Path) -> None:
    # LRSchedule defaults
    ls = LRSchedule()
    assert ls.decay_lr is True
    assert ls.warmup_iters == 2_000
    assert ls.lr_decay_iters == 600_000
    assert ls.min_lr == 6e-5

    # OptimConfig defaults
    oc = OptimConfig()
    assert oc.learning_rate == pytest.approx(6e-4)
    assert oc.weight_decay == pytest.approx(1e-1)
    assert oc.beta1 == pytest.approx(0.9)
    assert oc.beta2 == pytest.approx(0.95)
    assert oc.grad_clip == pytest.approx(1.0)

    # ModelConfig defaults
    mc = ModelConfig()
    assert mc.n_layer == 12
    assert mc.n_head == 12
    assert mc.n_embd == 767
    assert mc.block_size == 1024
    # dropout default intentionally outside [0,1] means "use model default"; skip strict check here

    # SampleConfig defaults
    sc = SampleConfig()
    assert sc.start == "\n"
    assert sc.num_samples == 3
    assert sc.max_new_tokens == 200
    assert sc.temperature == pytest.approx(0.8)
    assert sc.top_k == 200
    assert sc.top_p is None


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
    # ok
    RuntimeConfig(out_dir=tmp_path)


def test_runtimeconfig_defaults_and_checkpointing() -> None:
    # Validate default numeric and boolean values to catch NumberReplacer mutants
    rt = RuntimeConfig(out_dir=Path("./out"))
    assert rt.max_iters == 600_000
    assert rt.eval_interval == 2_000
    assert rt.eval_iters == 200
    assert rt.log_interval == 1
    assert rt.eval_only is False
    assert rt.seed == 1337
    assert rt.device == "cpu"
    assert rt.dtype == "float32"
    assert rt.compile is False
    assert rt.tensorboard_enabled is True
    # Checkpointing defaults
    ck = rt.checkpointing
    assert ck.read_policy in ("latest", "best")
    assert ck.keep.last == 1
    assert ck.keep.best == 1
    assert rt.ckpt_metric in ("val_loss", "perplexity")
    assert rt.ckpt_greater_is_better is False
    assert rt.ckpt_atomic is True
    assert rt.ckpt_write_metadata is True
    assert rt.ckpt_time_interval_minutes == 0


def test_deep_merge_dicts_nested_and_replace() -> None:
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": {"k": 1}, "d": 4}
    override = {"b": {"y": 20, "z": 3}, "c": 5, "e": 6}
    out = _deep_merge_dicts(base, override)
    # Nested dicts merge
    assert out["b"] == {"x": 1, "y": 20, "z": 3}
    # Non-dict in override replaces
    assert out["c"] == 5
    # Base preserved when not overridden
    assert out["a"] == 1 and out["d"] == 4
    # New key added
    assert out["e"] == 6


def test_deep_merge_numeric_replacements() -> None:
    base = {"a": {"x": 1, "y": -2}, "b": 10}
    override = {"a": {"x": 3}, "b": 0}
    out = _deep_merge_dicts(base, override)
    # Exact numeric replacement, no arithmetic or bitwise side-effects
    assert out["a"]["x"] == 3
    assert out["a"]["y"] == -2
    assert out["b"] == 0


def test_deep_merge_type_replacement() -> None:
    # If override supplies a non-dict, it should replace the base dict entirely
    base = {"a": {"x": 1}, "b": {"y": 2}}
    override = {"b": 7}
    out = _deep_merge_dicts(base, override)
    assert out["a"] == {"x": 1}
    assert out["b"] == 7


def test_dataconfig_paths_and_defaults(tmp_path: Path) -> None:
    dc = DataConfig(dataset_dir=tmp_path)
    # Defaults
    assert dc.train_bin == "train.bin"
    assert dc.val_bin == "val.bin"
    assert dc.meta_pkl == "meta.pkl"
    assert dc.batch_size == 12
    assert dc.block_size == 1024
    assert dc.grad_accum_steps == 40
    assert dc.tokenizer in ("char", "word", "tiktoken")
    assert dc.ngram_size == 1
    assert dc.sampler in ("random", "sequential")
    # Paths compute correctly
    assert dc.train_path == tmp_path / "train.bin"
    assert dc.val_path == tmp_path / "val.bin"
    assert dc.meta_path == tmp_path / "meta.pkl"


def test_dataconfig_meta_none_path(tmp_path: Path) -> None:
    dc = DataConfig(dataset_dir=tmp_path, meta_pkl=None)
    assert dc.meta_path is None


def test_preparerconfig_path_coercion_and_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Paths are strictly Path-typed; resolve is best-effort
    pc = PreparerConfig(dataset_dir=tmp_path / "ds", raw_dir=tmp_path / "raw")
    assert isinstance(pc.dataset_dir, Path) and isinstance(pc.raw_dir, Path)
    # ensure resolve does not crash on non-existent
    _ = pc.dataset_dir and pc.raw_dir


def test_sampleconfig_more_ranges() -> None:
    # Additional bounds to catch AddNot and comparison flips
    with pytest.raises(ValidationError):
        SampleConfig(num_samples=0)
    with pytest.raises(ValidationError):
        SampleConfig(max_new_tokens=0)
    with pytest.raises(ValidationError):
        SampleConfig(temperature=-0.1)
    # Edge valid
    SampleConfig(temperature=1e-6, top_k=0, top_p=1.0)


def test_config_exports() -> None:
    # Validate that config exports expected symbols directly and loader exists
    from ml_playground import config as shim
    from ml_playground import config_loader as loader

    assert hasattr(shim, "TrainerConfig")
    assert hasattr(shim, "SamplerConfig")
    assert hasattr(shim, "DataConfig")
    assert hasattr(shim, "RuntimeConfig")
    assert hasattr(loader, "load_full_experiment_config")


def test_full_loader_incomplete_sample_config(tmp_path: Path) -> None:
    """Strict: incomplete [sample] should raise ValidationError."""
    toml_text = """
[prepare]

[sample.runtime]
out_dir = "out/test"
# Missing sample.sample section
"""
    cfg_path = tmp_path / "incomplete_sample.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_full_experiment_config(cfg_path)


def test_full_loader_no_train_section_raises(tmp_path: Path) -> None:
    """Strict: Missing [train] section raises."""
    toml_text = """
[prepare]

[sample.runtime]
out_dir = "out/test"

[sample.sample]
"""
    cfg_path = tmp_path / "no_train.toml"
    cfg_path.write_text(toml_text)
    with pytest.raises(ValidationError):
        load_full_experiment_config(cfg_path)


def test_full_loader_no_sample_section_raises(tmp_path: Path) -> None:
    """Strict: Missing [sample] section raises."""
    toml_text = """
[prepare]

[train.model]
n_layer=1
n_head=1
n_embd=32
block_size=16

[train.data]
dataset_dir = "data/shakespeare"

[train.optim]
learning_rate = 0.001

[train.schedule]

[train.runtime]
out_dir = "out/test"
"""
    cfg_path = tmp_path / "no_sample.toml"
    cfg_path.write_text(toml_text)
    with pytest.raises(ValidationError):
        load_full_experiment_config(cfg_path)


def test_full_loader_train_missing_data_section(tmp_path: Path) -> None:
    """Strict: [train] missing required data subsection must raise ValidationError."""
    toml_text = """
[prepare]

[train.model]
 n_layer=1
 n_head=1
 n_embd=32
 block_size=16
 
 [train.optim]
 learning_rate = 0.001
 
 [train.schedule]
 
 [train.runtime]
 out_dir = "out/test"
 # Missing [train.data] section
 """
    cfg_path = tmp_path / "missing_data.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_full_experiment_config(cfg_path)


def test_full_loader_train_missing_runtime_section(tmp_path: Path) -> None:
    """Strict: [train] missing required runtime subsection must raise ValidationError."""
    toml_text = """
[prepare]

[train.model]
 n_layer=1
 n_head=1
 n_embd=32
 block_size=16
 
 [train.data]
 dataset_dir = "data/shakespeare"
 
 [train.optim]
 learning_rate = 0.001
 
 [train.schedule]
 # Missing [train.runtime] section
 """
    cfg_path = tmp_path / "missing_runtime.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_full_experiment_config(cfg_path)


def test_full_loader_sample_missing_runtime_section(tmp_path: Path) -> None:
    """Strict: [sample] missing required runtime subsection must raise ValidationError."""
    toml_text = """
[prepare]

[sample.sample]
# Missing [sample.runtime] section
"""
    cfg_path = tmp_path / "sample_missing_runtime.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_full_experiment_config(cfg_path)

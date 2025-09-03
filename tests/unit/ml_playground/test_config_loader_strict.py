from ml_playground.config_loader import load_sample_config_from_raw


def test_runtime_ref_only_train_runtime_merge_no_other_indirection() -> None:
    raw = {
        "train": {"runtime": {"out_dir": "./out_train", "seed": 123}},
        "sample": {
            "runtime_ref": "train.runtime",
            "runtime": {"seed": 999},
            "sample": {"start": "hi", "max_new_tokens": 1},
        },
    }
    cfg = load_sample_config_from_raw(raw, defaults_raw={})
    rt = cfg.runtime
    assert rt is not None
    # merged: seed overridden to 999, out_dir preserved from train.runtime
    assert rt.seed == 999
    # Path is relative and ends with out_train (do not depend on './' prefix)
    assert not rt.out_dir.is_absolute()
    assert rt.out_dir.name == "out_train"


def test_loader_does_not_rewrite_relative_paths() -> None:
    raw = {
        "sample": {
            "runtime": {
                "out_dir": "relative/out",
                "device": "cpu",
                "dtype": "float32",
                "seed": 1,
            },
            "sample": {"start": "hi", "max_new_tokens": 1},
        }
    }
    cfg = load_sample_config_from_raw(raw, defaults_raw={})
    assert cfg.runtime is not None
    # ensure the string is preserved exactly (no absolute resolution)
    assert str(cfg.runtime.out_dir) == "relative/out"

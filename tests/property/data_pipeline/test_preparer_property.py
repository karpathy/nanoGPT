from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import hypothesis.strategies as st
from hypothesis import HealthCheck, example, given, settings
import numpy as np
import pytest

from ml_playground.configuration.models import DataConfig, PreparerConfig, SharedConfig
from ml_playground.core.error_handling import DataError
from ml_playground.core.tokenizer import create_tokenizer
from ml_playground.data_pipeline.preparer import PreparationOutcome, create_pipeline


_TEXT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz \n", min_size=1, max_size=128)
_SPLITS = st.floats(
    min_value=0.1,
    max_value=0.9,
    allow_nan=False,
    allow_infinity=False,
)
_META_EXTRAS = st.dictionaries(
    keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6),
    values=st.integers(min_value=0, max_value=10),
    max_size=3,
)


def _shared_config(base: Path) -> SharedConfig:
    dataset_dir = base / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    return SharedConfig(
        experiment="pbt",
        config_path=base / "config.toml",
        project_home=base,
        dataset_dir=dataset_dir,
        train_out_dir=base / "train",
        sample_out_dir=base / "sample",
    )


@given(text=_TEXT, split=_SPLITS, meta=_META_EXTRAS)
@example(text="abc", split=0.5, meta={"mode": 1})
@settings(
    max_examples=20,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_prepare_pipeline_creates_artifacts(
    text: str, split: float, meta: dict[str, int]
) -> None:
    """`create_pipeline().prepare_from_text` yields deterministic artifacts matching metadata expectations."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = _shared_config(base)
        cfg = PreparerConfig(tokenizer_type="char", extras={"split": split})
        pipeline = create_pipeline(cfg, shared)

        tokenizer = create_tokenizer("char")
        outcome = pipeline.prepare_from_text(
            text,
            tokenizer,
            split=split,
            meta_extras=meta,
        )

        assert isinstance(outcome, PreparationOutcome)
        for file_path in outcome.created_files + outcome.updated_files:
            assert file_path.exists()
        assert not outcome.skipped_files

        train_expected = int(len(text) * split)
        val_expected = len(text) - train_expected
        assert outcome.metadata["train_tokens"] == train_expected
        assert outcome.metadata["val_tokens"] == val_expected
        for key, value in meta.items():
            assert outcome.metadata.get(key) == value

        train_arr = np.fromfile(shared.dataset_dir / "train.bin", dtype=np.uint16)
        val_arr = np.fromfile(shared.dataset_dir / "val.bin", dtype=np.uint16)
        assert train_arr.size == train_expected
        assert val_arr.size == val_expected


def test_prepare_pipeline_uses_data_config_paths() -> None:
    """Custom `DataConfig` extras should control output artifact locations."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = _shared_config(base)
        data_cfg = DataConfig(
            train_bin="train_custom.bin",
            val_bin="val_custom.bin",
            meta_pkl="meta_custom.pkl",
        )
        cfg = PreparerConfig(tokenizer_type="char", extras={"data_config": data_cfg})
        pipeline = create_pipeline(cfg, shared)

        tokenizer = create_tokenizer("char")
        pipeline.prepare_from_text("hello world", tokenizer)

        assert (shared.dataset_dir / "train_custom.bin").exists()
        assert (shared.dataset_dir / "val_custom.bin").exists()
        assert (shared.dataset_dir / "meta_custom.pkl").exists()


def test_prepare_pipeline_rejects_non_dataconfig_extra() -> None:
    """Providing a non-`DataConfig` `data_config` extra must raise `DataError`."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = _shared_config(base)
        cfg = PreparerConfig(tokenizer_type="char", extras={"data_config": "invalid"})
        pipeline = create_pipeline(cfg, shared)

        with pytest.raises(DataError, match="data_config"):
            pipeline.prepare_from_text("data", create_tokenizer("char"))


@pytest.mark.parametrize("raw_split", ["bad", -0.1, 1.5])
def test_prepare_pipeline_invalid_split_extra(raw_split: object) -> None:
    """Invalid `split` extras should raise `DataError` when no explicit split is provided."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = _shared_config(base)
        cfg = PreparerConfig(tokenizer_type="char", extras={"split": raw_split})
        pipeline = create_pipeline(cfg, shared)

        with pytest.raises(DataError, match="split ratio"):
            pipeline.prepare_from_text("sample", create_tokenizer("char"))

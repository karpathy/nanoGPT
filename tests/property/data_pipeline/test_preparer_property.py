from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings

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


@given(text=_TEXT, split=_SPLITS, meta=_META_EXTRAS)
@example(text="abc", split=0.5, meta={"mode": 1})
@settings(
    max_examples=20,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_prepare_pipeline_creates_artifacts(
    text: str,
    split: float,
    meta: dict[str, int],
    shared_config_factory: Callable[[Path], SharedConfig],
) -> None:
    """`create_pipeline().prepare_from_text` yields deterministic artifacts matching metadata expectations."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = shared_config_factory(base)
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


def test_prepare_pipeline_uses_data_config_paths(
    shared_config_factory: Callable[[Path], SharedConfig],
) -> None:
    """Custom `DataConfig` extras should control output artifact locations."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = shared_config_factory(base)
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


def test_prepare_pipeline_rejects_non_dataconfig_extra(
    shared_config_factory: Callable[[Path], SharedConfig],
) -> None:
    """Providing a non-`DataConfig` `data_config` extra must raise `DataError`."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = shared_config_factory(base)
        cfg = PreparerConfig(tokenizer_type="char", extras={"data_config": "invalid"})
        pipeline = create_pipeline(cfg, shared)

        with pytest.raises(DataError, match="data_config"):
            pipeline.prepare_from_text("data", create_tokenizer("char"))


@pytest.mark.parametrize("raw_split", ["bad", -0.1, 1.5])
def test_prepare_pipeline_invalid_split_extra(
    raw_split: object,
    shared_config_factory: Callable[[Path], SharedConfig],
) -> None:
    """Invalid `split` extras should raise `DataError` when no explicit split is provided."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        shared = shared_config_factory(base)
        cfg = PreparerConfig(tokenizer_type="char", extras={"split": raw_split})
        pipeline = create_pipeline(cfg, shared)

        with pytest.raises(DataError, match="split ratio"):
            pipeline.prepare_from_text("sample", create_tokenizer("char"))


def test_pipeline_run_reads_raw_text_path(
    tmp_path: Path,
    shared_config_factory: Callable[[Path], SharedConfig],
) -> None:
    base = tmp_path
    raw_dir = base / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "input.txt"
    raw_file.write_text("hello world", encoding="utf-8")

    shared = shared_config_factory(base)
    cfg = PreparerConfig(tokenizer_type="char", raw_text_path=raw_file)
    pipeline = create_pipeline(cfg, shared)

    outcome = pipeline.run()

    assert outcome.metadata["train_tokens"] > 0
    assert (shared.dataset_dir / "train.bin").exists()
    assert (shared.dataset_dir / "val.bin").exists()
    assert (shared.dataset_dir / "meta.pkl").exists()


def test_pipeline_run_uses_tokenizer_factory(
    tmp_path: Path,
    shared_config_factory: Callable[[Path], SharedConfig],
) -> None:
    base = tmp_path
    raw_file = base / "input.txt"
    raw_file.write_text("abc", encoding="utf-8")

    shared = shared_config_factory(base)
    calls: list[object] = []

    def _factory(kind: object) -> object:
        calls.append(kind)
        return create_tokenizer(kind)

    cfg = PreparerConfig(
        tokenizer_type="char",
        raw_text_path=raw_file,
        tokenizer_factory=_factory,
    )
    pipeline = create_pipeline(cfg, shared)
    pipeline.run()

    assert calls  # factory invoked
    assert (shared.dataset_dir / "meta.pkl").exists()


def test_pipeline_run_requires_raw_text_path(
    tmp_path: Path,
    shared_config_factory: Callable[[Path], SharedConfig],
) -> None:
    shared = shared_config_factory(tmp_path)
    cfg = PreparerConfig(tokenizer_type="char")
    pipeline = create_pipeline(cfg, shared)

    with pytest.raises(DataError, match="No raw text path"):
        pipeline.run()

from __future__ import annotations

from pathlib import Path


from ml_playground.experiments.bundestag_char.preparer import BundestagCharPreparer
from ml_playground.configuration import PreparerConfig


def test_preparer_allows_legacy_ngram_extra(tmp_path: Path) -> None:
    """Legacy ngram extras are now ignored, so preparation proceeds."""
    (tmp_path / "page1.txt").write_text("Hallo Bundestag", encoding="utf-8")
    cfg = PreparerConfig(
        extras={
            "ngram_size": 3,
            "dataset_dir_override": str(tmp_path),
        }
    )

    preparer = BundestagCharPreparer()
    report = preparer.prepare(cfg)

    ds_dir = tmp_path / "datasets"
    assert (ds_dir / "train.bin").exists()
    assert (ds_dir / "val.bin").exists()
    assert (ds_dir / "meta.pkl").exists()
    assert any("prepared dataset" in msg for msg in report.messages)


def test_preparer_dataset_override_uses_custom_base(tmp_path: Path) -> None:
    """Dataset artifacts are written under the provided dataset_dir_override."""
    base_dir = tmp_path / "custom"
    base_dir.mkdir()
    (base_dir / "page1.txt").write_text("Bundestag override", encoding="utf-8")

    cfg = PreparerConfig(extras={"dataset_dir_override": str(base_dir)})

    preparer = BundestagCharPreparer()
    report = preparer.prepare(cfg)

    ds_dir = base_dir / "datasets"
    assert (ds_dir / "train.bin").exists()
    assert (ds_dir / "val.bin").exists()
    assert (ds_dir / "meta.pkl").exists()
    assert report.created_files

from __future__ import annotations

from pathlib import Path
import pytest

from ml_playground import prepare as prep


def test_seed_text_file_copies_first_existing_candidate(tmp_path: Path):
    src1 = tmp_path / "a.txt"
    src2 = tmp_path / "b.txt"
    dst = tmp_path / "out" / "seed.txt"

    # Only src2 exists
    src2.write_text("hello", encoding="utf-8")

    prep.seed_text_file(dst, [src1, src2])
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "hello"


def test_seed_text_file_noop_if_dst_exists(tmp_path: Path):
    src = tmp_path / "in.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")

    prep.seed_text_file(dst, [src])
    # Should not overwrite existing dst
    assert dst.read_text(encoding="utf-8") == "old"


def test_seed_text_file_raises_when_no_candidates_exist(tmp_path: Path):
    dst = tmp_path / "dst.txt"
    with pytest.raises(FileNotFoundError):
        prep.seed_text_file(dst, [tmp_path / "missing1.txt", tmp_path / "missing2.txt"])

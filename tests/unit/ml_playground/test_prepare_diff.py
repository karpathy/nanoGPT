from __future__ import annotations

from pathlib import Path

from ml_playground import prepare as prep


def test_diff_files_updated_and_skipped(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    # create both files
    a.write_text("one", encoding="utf-8")
    b.write_text("keep", encoding="utf-8")

    before = prep.snapshot_files([a, b])

    # update a, leave b unchanged
    a.write_text("two", encoding="utf-8")

    created, updated, skipped = prep.diff_files([a, b], before)
    assert a in updated
    assert b in skipped

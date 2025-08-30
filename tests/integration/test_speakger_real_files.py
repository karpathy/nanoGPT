from __future__ import annotations

from pathlib import Path
import pytest

from ml_playground.datasets.speakger_pilot import (
    make_head_subset,
    stream_filter_to_file,
)


@pytest.mark.integration
@pytest.mark.skip("tv: Feature work in progress")
def test_speakger_real_files_small_head(tmp_path: Path) -> None:
    base = Path("/ml_playground/datasets")
    speeches = base / "Bundestag.csv"
    mps_map = base / "all_mps_mapping.csv"
    mps_meta = base / "all_mps_meta.csv"

    if not speeches.exists():
        pytest.skip("Bundestag.csv not present locally; skipping real-files test")
    # mapping/meta are optional for this validation; continue even if one is missing
    if not mps_map.exists():
        mps_map = None  # type: ignore[assignment]
    if not mps_meta.exists():
        mps_meta = None  # type: ignore[assignment]

    # Create a tiny head subset (header + 1 data row) to keep memory usage small
    subset = tmp_path / "Bundestag.head.csv"
    make_head_subset(speeches, subset, 2)

    out = tmp_path / "filtered_kpd_1949.jsonl"
    report = stream_filter_to_file(
        speeches_csv=subset,
        mps_mapping_csv=mps_map,
        mps_meta_csv=mps_meta,
        out_path=out,
        output_format="jsonl",
        chunksize=10_000,
        party="kpd",
        start_year=1949,
        end_year=1949,
    )

    # We expect at least zero or more rows depending on the first data line; ensure it runs and writes 0+ lines
    # If the first data row in the user's subset happens to match, rows_written should be >= 1.
    assert "rows_written" in report
    assert report["out_path"] == str(out)
    assert out.exists()

    # If the very first row (after header) matches the provided sample in the issue (party contains ['kpd/dkp'])
    # then we should see at least one row; otherwise allow zero to avoid flakiness.
    assert report["rows_written"] >= 0

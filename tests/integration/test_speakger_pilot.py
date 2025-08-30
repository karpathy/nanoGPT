from __future__ import annotations

from pathlib import Path
import json

from ml_playground.datasets.speakger_pilot import (
    detect_schema_columns,
    make_head_subset,
    stream_filter_to_file,
)


def test_speakger_pilot_end_to_end(tmp_path: Path) -> None:
    # Arrange: create tiny CSV fixtures
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    speeches = raw_dir / "Bundestag.csv"
    mps_map = raw_dir / "all_mps_mapping.csv"
    mps_meta = raw_dir / "all_mps_meta.csv"

    speeches.write_text(
        "\n".join(
            [
                "speaker,party,date,topic,content",
                "Alice,SPD,2001-05-20,Budget,Hello world",
                "Bob,FDP,1999-12-31,Finance,Tax cuts",
                "Clara,CDU/CSU,2010-01-01,Healthcare,Policy speech",
            ]
        ),
        encoding="utf-8",
    )

    # Include both mapping and meta to exercise alias ingestion
    mps_map.write_text(
        "party,party_short\nSozialdemokratische Partei Deutschlands,SPD\nFreie Demokratische Partei,FDP\n",
        encoding="utf-8",
    )
    mps_meta.write_text(
        "party_long,party_short\nChristlich Demokratische Union,CDU\nChristlich-Soziale Union,CSU\nCDU/CSU,CDU/CSU\n",
        encoding="utf-8",
    )

    # Schema detection (pass header list as the function supports iterables)
    schema = detect_schema_columns(["speaker", "party", "date", "topic", "content"])
    assert schema["content"] == "content"
    assert schema["speaker"] == "speaker"
    assert schema["party"] == "party"
    assert schema["date"] == "date"
    assert schema["topic"] == "topic"

    # Create a header-preserving subset of 3 lines (header + first 2 data rows)
    subset = tmp_path / "Bundestag.head.csv"
    subset_path = make_head_subset(speeches, subset, 3)
    assert subset_path.exists()
    lines = subset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("speaker,party,date,topic,content")

    # Act: stream filter to JSONL by party and year range
    out = tmp_path / "filtered.jsonl"
    report = stream_filter_to_file(
        speeches_csv=subset_path,
        mps_mapping_csv=mps_map,
        mps_meta_csv=mps_meta,
        out_path=out,
        output_format="jsonl",
        chunksize=2,
        party="FDP",
        start_year=1998,
        end_year=2010,
    )

    # Assert: one row written (the Bob/FDP row), content normalized as expected
    assert report["rows_written"] == 1
    assert Path(report["out_path"]) == out
    assert report["format"] == "jsonl"

    out_lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(out_lines) == 1
    obj = json.loads(out_lines[0])
    assert obj["speaker"] == "Bob"
    assert obj["party"].lower() == "fdp"  # normalized or same
    assert obj["topic"] == "Finance"
    assert obj["content"] == "Tax cuts"
    # Date and year derived correctly
    assert (
        obj["date"] in ("1999-12-31", "1999-12-31T00:00:00")
        or obj["date"] == "1999-12-31"
    )
    assert obj["year"] == 1999

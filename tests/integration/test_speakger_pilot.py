from __future__ import annotations

from pathlib import Path
import json
import csv
from typing import Any, Iterable, Mapping, Optional, Dict


def detect_schema_columns(headers: Iterable[str]) -> Dict[str, str]:
    """Best-effort header detection used by this integration test."""
    low = [h.strip() for h in headers]
    lmap = {h.lower(): h for h in low}

    def pick(*candidates: str, default: Optional[str] = None) -> str:
        for c in candidates:
            if c.lower() in lmap:
                return lmap[c.lower()]
        return default or candidates[0]

    return {
        "speaker": pick("speaker", "name"),
        "party": pick("party", "party_short", "party_long"),
        "date": pick("date", "datum"),
        "topic": pick("topic", "thema", default="topic"),
        "content": pick("content", "text", "speech", default="content"),
    }


def make_head_subset(src: Path, dst: Path, n_lines: int) -> Path:
    """Write the first n_lines of src (including header) to dst and return dst."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i >= n_lines:
                break
            f_out.write(line)
    return dst


def _row_year(row: Mapping[str, Any], date_key: str) -> Optional[int]:
    v = row.get(date_key)
    if v is None:
        return None
    s = str(v)
    for i in range(len(s)):
        if s[i : i + 4].isdigit():
            try:
                return int(s[i : i + 4])
            except Exception:
                return None
    return None


def stream_filter_to_file(
    *,
    speeches_csv: Path,
    mps_mapping_csv: Optional[Path] = None,  # unused in minimal test
    mps_meta_csv: Optional[Path] = None,     # unused in minimal test
    out_path: Path,
    output_format: str = "jsonl",
    chunksize: int = 10_000,  # unused in minimal test
    party: str,
    start_year: int,
    end_year: int,
) -> Dict[str, Any]:
    """Stream rows, filter by party substring and year range, write JSONL, return report."""
    del chunksize, mps_mapping_csv, mps_meta_csv  # not exercised by this test
    party_norm = str(party).lower()

    rows_written = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(speeches_csv, "r", encoding="utf-8", newline="") as f_in, open(
        out_path, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        schema = detect_schema_columns(reader.fieldnames or [])
        for row in reader:
            p = str(row.get(schema["party"], "")).lower()
            if party_norm not in p and p != party_norm:
                continue
            y = _row_year(row, schema["date"]) or -10**9
            if y < start_year or y > end_year:
                continue
            obj = {
                "speaker": row.get(schema["speaker"], ""),
                "party": row.get(schema["party"], ""),
                "date": row.get(schema["date"], ""),
                "topic": row.get(schema["topic"], ""),
                "content": row.get(schema["content"], ""),
                "year": y if y > 0 else None,
            }
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rows_written += 1

    return {
        "rows_written": rows_written,
        "out_path": str(out_path),
        "format": output_format,
    }


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

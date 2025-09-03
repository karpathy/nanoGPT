from __future__ import annotations
from pathlib import Path


from ml_playground.analysis.sample_quality import (
    _extract_header,
    _line_stats,
    _ngram_stats,
    _find_anomalies,
    analyze_sample_text,
    analyze_sample_file,
    format_analysis,
)


def test_extract_header_variants():
    lines = [
        "Sprecher: Dr. Example",
        "Thema: Haushalt",
        "Jahr: 2021",
        "Jahr: 2022",
        "Some body line",
    ]
    h = _extract_header(lines)
    assert h.speaker == "Dr. Example"
    assert h.topic == "Haushalt"
    assert h.year == "2021"
    assert h.year_count == 2

    lines2 = ["no header here"]
    h2 = _extract_header(lines2)
    assert (
        h2.speaker is None
        and h2.topic is None
        and h2.year is None
        and h2.year_count == 0
    )


def test_line_stats_repeats_and_runs():
    lines = [
        "a",
        "a",
        "b",
        "",
        "b",
        "b",
        "c",
        "c",
        "c",
    ]
    ls = _line_stats(lines)
    assert ls.total_lines == len(lines)
    assert ls.non_empty_lines == 8
    assert ls.longest_identical_run == 3
    assert any(s == "b" and c == 3 for s, c in ls.top_repeated_lines)


def test_ngram_stats_and_fallback():
    lines = ["Hello, world! Hello."]
    ns = _ngram_stats(lines, 1)
    assert ns.n == 1
    assert ns.unique_ngrams > 0

    ns2 = _ngram_stats(lines, 0)  # fallback to 3
    assert ns2.n == 3


def test_find_anomalies_trailing_and_years():
    lines = [
        "Some intro.",
        "Jahr: 2020",
        "Body mentions 2018 and 1999 too",
        "unfinished line with year 2025",
    ]
    an = _find_anomalies(lines)
    assert an.trailing_incomplete_line is True
    assert (
        "2018" in an.stray_year_tokens
        and "1999" in an.stray_year_tokens
        and "2025" in an.stray_year_tokens
    )


def test_analyze_and_format_text_and_file(tmp_path: Path):
    text = "\n".join(
        [
            "Sprecher: A",
            "Thema: B",
            "Jahr: 2022",
            "Hello world.",
            "Hello world.",
            "Bye.",
        ]
    )
    a = analyze_sample_text(text, ngram_n=2)
    s = format_analysis(a)
    assert (
        "== Header ==" in s
        and "== Lines ==" in s
        and "== N-grams ==" in s
        and "== Anomalies ==" in s
    )

    p = tmp_path / "sample.txt"
    p.write_text(text, encoding="utf-8")
    a2 = analyze_sample_file(p, ngram_n=2)
    assert a2.header.speaker == "A"

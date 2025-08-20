from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, Iterator
import re


HEADER_KEYS = ("Sprecher:", "Thema:", "Jahr:")


@dataclass(frozen=True)
class Header:
    speaker: str | None
    topic: str | None
    year: str | None
    year_count: int


@dataclass(frozen=True)
class LineStats:
    total_lines: int
    non_empty_lines: int
    unique_lines: int
    unique_ratio: float
    longest_identical_run: int
    top_repeated_lines: list[tuple[str, int]]


@dataclass(frozen=True)
class NgramStats:
    n: int
    unique_ngrams: int
    top_repeated_ngrams: list[tuple[str, int]]


@dataclass(frozen=True)
class Anomalies:
    trailing_incomplete_line: bool
    stray_year_tokens: list[str]


@dataclass(frozen=True)
class SampleAnalysis:
    header: Header
    lines: LineStats
    ngrams: NgramStats
    anomalies: Anomalies


def _iter_tokens(lines: Iterable[str]) -> Iterator[str]:
    token_re = re.compile(r"\w+|[^\w\s]")
    for line in lines:
        for tok in token_re.findall(line):
            yield tok


def _extract_header(lines: list[str]) -> Header:
    speaker: str | None = None
    topic: str | None = None
    year: str | None = None
    year_count = 0
    for line in lines[:10]:  # header usually in first few lines
        s = line.strip()
        if s.startswith("Sprecher:") and speaker is None:
            speaker = s.split("Sprecher:", 1)[1].strip() or None
        elif s.startswith("Thema:") and topic is None:
            topic = s.split("Thema:", 1)[1].strip() or None
        elif s.startswith("Jahr:"):
            year_count += 1
            if year is None:
                year = s.split("Jahr:", 1)[1].strip() or None
    return Header(speaker=speaker, topic=topic, year=year, year_count=year_count)


def _line_stats(lines: list[str]) -> LineStats:
    non_empty = [ln for ln in lines if ln.strip()]
    total_lines = len(lines)
    non_empty_lines = len(non_empty)
    counts = Counter(non_empty)
    unique_lines = len(counts)
    unique_ratio = (unique_lines / non_empty_lines) if non_empty_lines else 0.0

    # Longest consecutive identical-line run
    longest = 0
    current = 0
    last: str | None = None
    for ln in non_empty:
        if ln == last:
            current += 1
        else:
            current = 1
            last = ln
        if current > longest:
            longest = current

    top_repeated_lines = [(s, c) for s, c in counts.most_common(10) if c > 1]
    return LineStats(
        total_lines=total_lines,
        non_empty_lines=non_empty_lines,
        unique_lines=unique_lines,
        unique_ratio=unique_ratio,
        longest_identical_run=longest,
        top_repeated_lines=top_repeated_lines,
    )


def _ngram_stats(lines: list[str], n: int) -> NgramStats:
    # Tokenize and compute n-gram counts
    tokens = list(_iter_tokens(lines))
    if n <= 0:
        n = 3
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] += 1
    # Prepare top repeated ngrams (joined by space) and counts
    top = [
        (" ".join(g), c) for g, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15] if c > 1
    ]
    return NgramStats(n=n, unique_ngrams=len(counts), top_repeated_ngrams=top)


def _find_anomalies(lines: list[str]) -> Anomalies:
    trailing_incomplete = False
    if lines:
        last = lines[-1]
        # Heuristic: a very short final line without punctuation may be truncated
        trailing_incomplete = bool(last.strip()) and not re.search(r"[.!?…]$", last.strip())

    # Look for standalone 4-digit years in body (not header lines)
    year_pat = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
    stray: list[str] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("Jahr:"):
            continue
        for m in year_pat.findall(s):
            # m comes from grouped regex; we want the full year. Adjust:
            for full in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", s):
                stray.append(full)
            break
    return Anomalies(trailing_incomplete_line=trailing_incomplete, stray_year_tokens=sorted(set(stray)))


def analyze_sample_text(text: str, ngram_n: int = 3) -> SampleAnalysis:
    lines = text.splitlines()
    header = _extract_header(lines)
    lstats = _line_stats(lines)
    nstats = _ngram_stats(lines, ngram_n)
    anomalies = _find_anomalies(lines)
    return SampleAnalysis(header=header, lines=lstats, ngrams=nstats, anomalies=anomalies)


def analyze_sample_file(path: Path, ngram_n: int = 3) -> SampleAnalysis:
    text = path.read_text(encoding="utf-8")
    return analyze_sample_text(text, ngram_n)


def format_analysis(analysis: SampleAnalysis) -> str:
    h = analysis.header
    ls = analysis.lines
    ng = analysis.ngrams
    an = analysis.anomalies
    parts: list[str] = []
    parts.append("== Header ==")
    parts.append(f"Sprecher: {h.speaker or '-'}")
    parts.append(f"Thema:    {h.topic or '-'}")
    parts.append(f"Jahr:     {h.year or '-'} (occurrences: {h.year_count})")
    parts.append("")

    parts.append("== Lines ==")
    parts.append(
        f"lines: total={ls.total_lines}, non_empty={ls.non_empty_lines}, unique={ls.unique_lines}, unique_ratio={ls.unique_ratio:.2f}"
    )
    parts.append(f"longest_identical_run: {ls.longest_identical_run}")
    if ls.top_repeated_lines:
        parts.append("top_repeated_lines:")
        for s, c in ls.top_repeated_lines:
            preview = s.strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."
            parts.append(f"  - ({c}x) {preview}")
    else:
        parts.append("top_repeated_lines: -")
    parts.append("")

    parts.append("== N-grams ==")
    parts.append(f"n={ng.n}, unique_ngrams={ng.unique_ngrams}")
    if ng.top_repeated_ngrams:
        parts.append("top_repeated_ngrams:")
        for g, c in ng.top_repeated_ngrams[:10]:
            preview = g
            if len(preview) > 80:
                preview = preview[:77] + "..."
            parts.append(f"  - ({c}x) {preview}")
    else:
        parts.append("top_repeated_ngrams: -")
    parts.append("")

    parts.append("== Anomalies ==")
    parts.append(f"trailing_incomplete_line: {an.trailing_incomplete_line}")
    if an.stray_year_tokens:
        parts.append(f"stray_year_tokens: {', '.join(an.stray_year_tokens)}")
    else:
        parts.append("stray_year_tokens: -")

    return "\n".join(parts)

#!/usr/bin/env python3
"""Generate coverage badges for line and branch metrics."""

from __future__ import annotations

import html
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Final

from coverage import Coverage
from coverage.results import display_covered

BADGE_LEFT_COLOR: Final[str] = "#555555"
BADGE_HEIGHT: Final[int] = 20
LEFT_SECTION_WIDTH: Final[float] = 87.5
RIGHT_SECTION_WIDTH: Final[float] = 52.0
BADGE_WIDTH: Final[float] = LEFT_SECTION_WIDTH + RIGHT_SECTION_WIDTH

COLOR_THRESHOLDS: Final[tuple[tuple[float, str], ...]] = (
    (95.0, "#0e8a16"),  # green
    (90.0, "#2cbe4e"),  # bright green
    (80.0, "#9be9a8"),  # light green
    (70.0, "#dfb317"),  # yellow
    (60.0, "#ffa500"),  # orange
    (0.0, "#e05d44"),  # red
)

SVG_TEMPLATE: Final[str] = (
    """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" role=\"img\" aria-label=\"{aria_label}\">\n  <title>{title}</title>\n  <linearGradient id=\"smooth\" x2=\"0\" y2=\"100%\"><stop offset=\"0\" stop-color=\"#bbb\" stop-opacity=\".1\"/><stop offset=\"1\" stop-opacity=\".1\"/></linearGradient>\n  <clipPath id=\"round\"><rect width=\"{width}\" height=\"{height}\" rx=\"3\" fill=\"#fff\"/></clipPath>\n  <g clip-path=\"url(#round)\">\n    <rect width=\"{left_width}\" height=\"{height}\" fill=\"{left_color}\"/>\n    <rect x=\"{left_width}\" width=\"{right_width}\" height=\"{height}\" fill=\"{right_color}\"/>\n    <rect width=\"{width}\" height=\"{height}\" fill=\"url(#smooth)\"/>\n  </g>\n  <g fill=\"#fff\" text-anchor=\"middle\" font-family=\"DejaVu Sans,Verdana,Geneva,sans-serif\" font-size=\"11\">\n    <text x=\"{label_x}\" y=\"{text_y}\" dominant-baseline=\"middle\">{label_text}</text>\n    <text x=\"{value_x}\" y=\"{text_y}\" dominant-baseline=\"middle\">{value_text}</text>\n  </g>\n</svg>\n"""
)


def color_for(percentage: float) -> str:
    for threshold, color in COLOR_THRESHOLDS:
        if percentage >= threshold:
            return color
    return COLOR_THRESHOLDS[-1][1]


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def percent_display(value: float, precision: int) -> tuple[float, str]:
    display = display_covered(value, precision)
    return float(display.rstrip("%")), display


def compute_branch_percent(totals: dict[str, object]) -> float:
    covered = _to_float(totals.get("covered_branches"))
    total = _to_float(totals.get("num_branches"))
    if total <= 0.0:
        return 100.0
    return (covered / total) * 100.0


def generate_badge(label: str, percentage: float, display: str) -> str:
    label_text = f"{label} coverage"
    value_text = display
    aria_label = f"{label_text}: {value_text}"
    return SVG_TEMPLATE.format(
        width=f"{BADGE_WIDTH:.1f}",
        height=f"{BADGE_HEIGHT}",
        left_width=f"{LEFT_SECTION_WIDTH:.1f}",
        right_width=f"{RIGHT_SECTION_WIDTH:.1f}",
        left_color=BADGE_LEFT_COLOR,
        right_color=color_for(percentage),
        label_x=f"{LEFT_SECTION_WIDTH / 2:.1f}",
        value_x=f"{LEFT_SECTION_WIDTH + (RIGHT_SECTION_WIDTH / 2):.1f}",
        text_y=f"{BADGE_HEIGHT / 2:.1f}",
        label_text=html.escape(label_text),
        value_text=html.escape(value_text),
        aria_label=html.escape(aria_label),
        title=html.escape(aria_label),
    )


def write_badge(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")


def main(args: list[str]) -> int:
    if len(args) != 3:
        print(
            "Usage: coverage_badges.py <coverage-json> <output-dir>",
            file=sys.stderr,
        )
        return 2

    coverage_json = Path(args[1]).resolve()
    output_dir = Path(args[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    totals = json.loads(coverage_json.read_text(encoding="utf-8"))["totals"]
    coverage_db = coverage_json.with_name("coverage.sqlite")

    coverage_obj = Coverage(data_file=str(coverage_db))
    coverage_obj.load()
    buffer = StringIO()
    total_percentage = coverage_obj.report(file=buffer)
    precision = coverage_obj.config.precision

    line_percentage, line_display = percent_display(total_percentage, precision)
    branch_percentage_raw = compute_branch_percent(totals)
    branch_percentage, branch_display = percent_display(
        branch_percentage_raw, precision
    )

    badges = (
        ("coverage-lines.svg", generate_badge("Line", line_percentage, line_display)),
        (
            "coverage-branches.svg",
            generate_badge("Branch", branch_percentage, branch_display),
        ),
        ("coverage.svg", generate_badge("Line", line_percentage, line_display)),
    )

    for filename, svg in badges:
        write_badge(output_dir / filename, svg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

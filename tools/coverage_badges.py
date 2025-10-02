#!/usr/bin/env python3
"""Generate coverage badges for line and branch metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final

from pybadges import badge

BADGE_LEFT_COLOR: Final[str] = "#555555"

COLOR_THRESHOLDS: Final[tuple[tuple[float, str], ...]] = (
    (95.0, "#0e8a16"),  # green
    (90.0, "#2cbe4e"),  # bright green
    (80.0, "#9be9a8"),  # light green
    (70.0, "#dfb317"),  # yellow
    (60.0, "#ffa500"),  # orange
    (0.0, "#e05d44"),  # red
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


def percent_display(value: float) -> str:
    return f"{value:.2f}%"


def compute_branch_percent(totals: dict[str, object]) -> float:
    covered = _to_float(totals.get("covered_branches"))
    total = _to_float(totals.get("num_branches"))
    if total <= 0.0:
        return 100.0
    return (covered / total) * 100.0


def generate_badge(label: str, percentage: float) -> str:
    return badge(
        left_text=f"{label} coverage",
        right_text=percent_display(percentage),
        left_color=BADGE_LEFT_COLOR,
        right_color=color_for(percentage),
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
    line_percentage = _to_float(totals.get("percent_covered"))
    branch_percentage = compute_branch_percent(totals)

    badges = (
        ("coverage-lines.svg", generate_badge("Line", line_percentage)),
        ("coverage-branches.svg", generate_badge("Branch", branch_percentage)),
        ("coverage.svg", generate_badge("Line", line_percentage)),
    )

    for filename, svg in badges:
        write_badge(output_dir / filename, svg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = [
#   "typer>=0.12.3",
# ]
# ///
"""Linting and static analysis commands for ml_playground."""

from __future__ import annotations

import typer

from tools import task_utils as utils

app = typer.Typer(help="Linting and static analysis commands.", no_args_is_help=True)


@app.command()
def lint() -> None:
    """Run Ruff lint checks (with autofix disabled)."""
    utils.uv_run("ruff", "check", ".")


@app.command()
def format() -> None:
    """Auto-fix and format code with Ruff."""
    utils.uv_run("ruff", "check", "--fix", ".")
    utils.uv_run("ruff", "format", ".")


@app.command("lint-check")
def lint_check() -> None:
    """Run Ruff in check-only mode."""
    utils.uv_run("ruff", "check", ".")


@app.command()
def deadcode() -> None:
    """Scan for dead code using vulture."""
    utils.uv_run("vulture", utils.PKG, "--min-confidence", "90")


@app.command()
def pyright() -> None:
    """Run Pyright type checks."""
    utils.uv_run("pyright", utils.PKG)


@app.command()
def mypy() -> None:
    """Run Mypy type checks."""
    utils.uv_run("mypy", "--incremental", utils.PKG)


@app.command()
def typecheck() -> None:
    """Run both Pyright and Mypy."""
    pyright()
    mypy()


def main() -> None:  # pragma: no cover
    try:
        app()
    except utils.CommandError as exc:  # pragma: no cover
        raise typer.Exit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()

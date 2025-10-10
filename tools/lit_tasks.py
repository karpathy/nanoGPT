#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = [
#   "typer>=0.12.3",
# ]
# ///
"""LIT demo environment helpers for ml_playground."""

from __future__ import annotations

import typer

from tools import task_utils as utils

app = typer.Typer(
    help="Manage the optional LIT demo environment.", no_args_is_help=True
)


@app.command()
def setup(
    python_version: str = typer.Option(
        "3.12", "--python-version", help="Python version for the isolated venv"
    ),
    recreate: bool = typer.Option(
        False, "--recreate", help="Recreate the LIT virtual environment"
    ),
) -> None:
    """Create a dedicated Python environment for the LIT demo."""
    if recreate and utils.LIT_VENV.exists():
        typer.echo(f"Removing {utils.LIT_VENV}")
        utils.remove_path(utils.LIT_VENV)
    utils.uv("venv", "--python", python_version, str(utils.LIT_VENV))
    utils.uv(
        "pip",
        "install",
        "-p",
        str(utils.lit_python()),
        "-r",
        str(utils.LIT_REQUIREMENTS),
    )


@app.command()
def run(
    port: int = typer.Option(5432, "--port", help="Port to bind the LIT server"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    open_browser: bool = typer.Option(
        False, "--open-browser", help="Open the browser automatically"
    ),
) -> None:
    """Start the minimal LIT demo server."""
    python_bin = utils.lit_python()
    if not python_bin.exists():
        raise typer.BadParameter("LIT environment missing; run 'setup' first")
    args = [
        "python",
        "-m",
        "ml_playground.analysis.lit.integration",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if open_browser:
        args.append("--open-browser")
    utils.uv_run(*args, python=str(python_bin), no_project=True)


@app.command()
def stop(
    port: int = typer.Option(5432, "--port", help="Port to terminate"),
    graceful: bool = typer.Option(
        True, "--graceful/--force", help="Attempt graceful shutdown"
    ),
) -> None:
    """Stop the LIT demo server bound to the given port."""
    python_bin = utils.lit_python()
    if not python_bin.exists():
        typer.echo("[info] LIT environment not found; nothing to stop")
        return
    args = ["python", "tools/port_kill.py", "--port", str(port)]
    if graceful:
        args.append("--graceful")
    utils.uv_run(*args, python=str(python_bin), no_project=True)


def main() -> None:  # pragma: no cover
    try:
        app()
    except utils.CommandError as exc:  # pragma: no cover
        raise typer.Exit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()

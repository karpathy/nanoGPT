#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = [
#   "typer>=0.12.3",
# ]
# ///
"""Developer environment management commands for ml_playground."""

from __future__ import annotations

from pathlib import Path

import typer

from tools import task_utils as utils

app = typer.Typer(
    help="Environment and developer tooling commands.", no_args_is_help=True
)


@app.command()
def setup(
    clear: bool = typer.Option(
        False, "--clear", help="Remove existing virtual env first"
    ),
) -> None:
    """Create a fresh uv-managed virtual environment and install all deps."""
    if clear:
        utils.uv("venv", "--clear")
    utils.uv("sync", "--all-groups")


@app.command()
def sync(
    groups: list[str] = typer.Option(
        None,
        "--group",
        help="Sync the specified dependency groups (repeatable).",
    ),
    all_groups: bool = typer.Option(
        False,
        "--all-groups",
        help="Install all optional dependency groups.",
    ),
) -> None:
    """Sync project dependencies."""
    args = ["sync"]
    if all_groups:
        args.append("--all-groups")
    elif groups:
        for group in groups:
            args.extend(["--group", group])
    utils.uv(*args)


@app.command()
def verify() -> None:
    """Ensure the project package imports correctly."""
    utils.uv_run(
        "python", "-c", f"import {utils.PKG}; print('✓ {utils.PKG} import OK')"
    )


@app.command()
def clean() -> None:
    """Remove caches and temporary build artifacts."""
    for path in [
        utils.ROOT / ".pytest_cache",
        utils.ROOT / ".mypy_cache",
        utils.ROOT / ".ruff_cache",
        utils.ROOT / "htmlcov",
    ]:
        utils.remove_path(path)
    for pycache in utils.ROOT.rglob("__pycache__"):
        utils.remove_path(pycache)


@app.command("ai-guidelines")
def ai_guidelines(
    tool: str = typer.Argument(..., help="Target tool name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions"),
) -> None:
    """Set up AI guideline symlinks for the requested tool."""
    command = ["python", "tools/setup_ai_guidelines.py", tool]
    if dry_run:
        command.append("--dry-run")
    utils.uv_run(*command)


@app.command()
def tensorboard(
    logdir: Path = typer.Option(
        ...,
        "--logdir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="TensorBoard log directory",
    ),
    port: int = typer.Option(6006, "--port", help="Port to bind"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface"),
) -> None:
    """Launch TensorBoard for the given log directory."""
    utils.uv_run(
        "tensorboard", "--logdir", str(logdir), "--port", str(port), "--host", host
    )


@app.command("gguf-help")
def gguf_help() -> None:
    """Show llama.cpp GGUF conversion help."""
    try:
        utils.uv_run("python", "tools/llama_cpp/convert-hf-to-gguf.py", "--help")
    except utils.CommandError:
        typer.echo("[info] GGUF converter exited with a non-zero status", err=True)


def main() -> None:  # pragma: no cover
    try:
        app()
    except utils.CommandError as exc:  # pragma: no cover
        raise typer.Exit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()

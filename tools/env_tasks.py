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
lit_app = typer.Typer(help="Manage the optional LIT demo environment")
app.add_typer(lit_app, name="lit")


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
    all_groups: bool = typer.Option(
        True, "--all-groups/--project-only", help="Install dev dependencies (default)"
    ),
) -> None:
    """Sync project dependencies."""
    args = ["sync"]
    if all_groups:
        args.append("--all-groups")
    utils.uv(*args)


@app.command()
def verify() -> None:
    """Ensure the project package imports correctly."""
    utils.uv_run(
        "python", "-c", f"import {utils.PKG}; print('✓ {utils.PKG} import OK')"
    )


@app.command()
def lint() -> None:
    """Run Ruff lint checks."""
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


@lit_app.command("setup")
def lit_setup(
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


@lit_app.command("run")
def lit_run(
    port: int = typer.Option(5432, "--port", help="Port to bind the LIT server"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    open_browser: bool = typer.Option(
        False, "--open-browser", help="Open the browser automatically"
    ),
) -> None:
    """Start the minimal LIT demo server."""
    python_bin = utils.lit_python()
    if not python_bin.exists():
        raise typer.BadParameter("LIT environment missing; run 'lit setup' first")
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


@lit_app.command("stop")
def lit_stop(
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

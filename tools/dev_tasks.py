#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = [
#   "typer>=0.12.3",
# ]
# ///
"""Developer workflow commands powered by uvx.

This CLI replaces the historical Makefile targets and mirrors the
underlying behavior using uv-managed environments.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer


def _discover_root() -> Path:
    """Find the repository root by locating pyproject.toml."""

    def expand_chain(start: Path, seen: set[Path], order: list[Path]) -> None:
        for path in (start, *start.parents):
            if path in seen:
                continue
            seen.add(path)
            order.append(path)

    seen: set[Path] = set()
    ordered: list[Path] = []
    expand_chain(Path(__file__).resolve(), seen, ordered)
    expand_chain(Path.cwd(), seen, ordered)

    for path in ordered:
        if (path / "pyproject.toml").exists():
            return path

    return Path(__file__).resolve().parents[1]


ROOT = _discover_root()
PKG = "ml_playground"
PYTEST_BASE = ["-q", "-n", "auto", "-W", "error", "--strict-markers", "--strict-config"]
PRE_COMMIT_CONFIG = ROOT / ".githooks" / ".pre-commit-config.yaml"
CACHE_DIR = ROOT / ".cache"
LIT_VENV = ROOT / ".venv312"
LIT_REQUIREMENTS = ROOT / "ml_playground" / "analysis" / "lit" / "requirements.txt"

app = typer.Typer(
    help="Developer utility commands executed via uvx.", no_args_is_help=True
)
mutation_app = typer.Typer(help="Mutation testing helpers")
lit_app = typer.Typer(help="Manage the optional LIT demo environment")
app.add_typer(mutation_app, name="mutation")
app.add_typer(lit_app, name="lit")


class CommandError(RuntimeError):
    """Raised when an invoked subprocess exits with a failure."""


def _echo_command(command: List[str]) -> None:
    formatted = " ".join(shlex.quote(arg) for arg in command)
    typer.echo(f"$ {formatted}")


def _run(
    command: List[str], *, env: Optional[dict[str, str]] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Execute *command* inside the project root."""
    _echo_command(command)
    result = subprocess.run(command, cwd=ROOT, env=env)
    if check and result.returncode != 0:
        raise CommandError(f"Command failed with exit code {result.returncode}")
    return result


def uv(
    *args: str, env: Optional[dict[str, str]] = None, check: bool = True
) -> subprocess.CompletedProcess:
    return _run(["uv", *args], env=env, check=check)


def uv_run(
    *args: str,
    group_dev: bool = True,
    python: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    no_project: bool = False,
) -> subprocess.CompletedProcess:
    command: List[str] = ["uv", "run"]
    command.extend(["--project", str(ROOT)])
    if python:
        command.extend(["--python", python])
    if no_project:
        command.append("--no-project")
    command.extend(args)
    return _run(command, env=env, check=check)


def _ensure_cache_dirs() -> None:
    for subdir in ["coverage", "hypothesis", "uv"]:
        (CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def _session_file() -> Path:
    return CACHE_DIR / "cosmic-ray" / "session.sqlite"


def _lit_python() -> Path:
    if sys.platform.startswith("win"):
        return LIT_VENV / "Scripts" / "python.exe"
    return LIT_VENV / "bin" / "python"


def _pytest(extra: Optional[List[str]] = None) -> List[str]:
    return ["pytest", *PYTEST_BASE, *(extra or [])]


def _forwarded_args(args: Optional[List[str]]) -> List[str]:
    return list(args or [])


@app.command()
def setup() -> None:
    """Create a fresh uv-managed virtual environment and install all deps."""
    uv("venv", "--clear")
    uv("sync", "--all-groups")


@app.command()
def sync() -> None:
    """Sync project and development dependencies."""
    uv("sync", "--all-groups")


@app.command()
def verify() -> None:
    """Ensure the project package imports correctly."""
    uv_run("python", "-c", f"import {PKG}; print('✓ {PKG} import OK')", group_dev=False)


@app.command("pytest")
def pytest_core(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Invoke pytest with the shared configuration."""
    uv_run(*_pytest(_forwarded_args(args)))


@app.command()
def test(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run the full test suite."""
    uv_run(*_pytest(["tests", *_forwarded_args(args)]))


@app.command()
def unit(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run unit tests."""
    uv_run(*_pytest(["tests/unit", *_forwarded_args(args)]))


@app.command("property")
def property_tests(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run property-based tests."""
    uv_run(*_pytest(["tests/property", *_forwarded_args(args)]))


@app.command("unit-cov")
def unit_cov(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run unit tests with coverage reporting."""
    uv_run(
        *_pytest(
            [
                f"--cov={PKG}",
                "--cov-report=term-missing",
                "tests/unit",
                *_forwarded_args(args),
            ]
        )
    )


@app.command()
def integration(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run integration tests."""
    uv_run(*_pytest(["-m", "integration", "--no-cov", *_forwarded_args(args)]))


@app.command()
def e2e(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run end-to-end tests."""
    uv_run(*_pytest(["tests/e2e", *_forwarded_args(args)]))


@app.command()
def acceptance(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run acceptance tests."""
    uv_run(*_pytest(["tests/acceptance", *_forwarded_args(args)]))


@app.command()
def quality() -> None:
    """Run the full pre-commit quality gate."""
    uv_run("pre-commit", "run", "--config", str(PRE_COMMIT_CONFIG), "--all-files")


@app.command("quality-fast")
def quality_fast() -> None:
    """Run lint- and format-only pre-commit hooks."""
    uv_run(
        "pre-commit", "run", "--config", str(PRE_COMMIT_CONFIG), "--all-files", "ruff"
    )
    uv_run(
        "pre-commit",
        "run",
        "--config",
        str(PRE_COMMIT_CONFIG),
        "--all-files",
        "ruff-format",
    )
    uv_run(
        "pre-commit",
        "run",
        "--config",
        str(PRE_COMMIT_CONFIG),
        "--all-files",
        "mdformat",
    )


@app.command()
def lint() -> None:
    """Lint the codebase with Ruff."""
    uv_run("ruff", "check", ".")


@app.command()
def format() -> None:
    """Auto-fix and format code with Ruff."""
    uv_run("ruff", "check", "--fix", ".")
    uv_run("ruff", "format", ".")


@app.command("lint-check")
def lint_check() -> None:
    """Run Ruff in check-only mode."""
    uv_run("ruff", "check", ".")


@app.command()
def deadcode() -> None:
    """Scan for dead code using vulture."""
    uv_run("vulture", PKG, "--min-confidence", "90")


@app.command()
def pyright() -> None:
    """Run Pyright type checks."""
    uv_run("pyright", PKG)


@app.command()
def mypy() -> None:
    """Run Mypy type checks."""
    uv_run("mypy", "--incremental", PKG)


@app.command()
def typecheck() -> None:
    """Run both Pyright and Mypy."""
    pyright()
    mypy()


@app.command()
def clean() -> None:
    """Remove caches and temporary build artifacts."""
    for path in [
        ROOT / ".pytest_cache",
        ROOT / ".mypy_cache",
        ROOT / ".ruff_cache",
        ROOT / "htmlcov",
    ]:
        if path.exists():
            typer.echo(f"Removing {path}")
            shutil.rmtree(path, ignore_errors=True)
    for pycache in ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def _cli_command(subcommand: str, exp: str, config: Optional[Path]) -> List[str]:
    command = ["python", "-m", f"{PKG}.cli", subcommand, exp]
    if config:
        command.extend(["--exp-config", str(config)])
    return command


@app.command()
def prepare(
    exp: str = typer.Argument(..., help="Experiment name"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Optional TOML config path",
    ),
) -> None:
    """Prepare dataset artifacts for an experiment."""
    uv_run(*_cli_command("prepare", exp, config), group_dev=False)


@app.command()
def train(
    exp: str = typer.Argument(..., help="Experiment name"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Optional TOML config path",
    ),
) -> None:
    """Train a model for the specified experiment."""
    uv_run(*_cli_command("train", exp, config), group_dev=False)


@app.command()
def sample(
    exp: str = typer.Argument(..., help="Experiment name"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Optional TOML config path",
    ),
) -> None:
    """Sample from a trained model."""
    uv_run(*_cli_command("sample", exp, config), group_dev=False)


@app.command()
def loop(
    exp: str = typer.Argument(..., help="Experiment name"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Optional TOML config path",
    ),
) -> None:
    """Execute the full prepare/train/sample loop."""
    uv_run(*_cli_command("loop", exp, config), group_dev=False)


@app.command("ai-guidelines")
def ai_guidelines(
    tool: str = typer.Argument(..., help="Target tool name"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview actions without modifying files"
    ),
) -> None:
    """Set up AI guidelines symlinks for the requested tool."""
    command = ["python", "tools/setup_ai_guidelines.py", tool]
    if dry_run:
        command.append("--dry-run")
    uv_run(*command)


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
) -> None:
    """Launch TensorBoard for the given log directory."""
    uv_run("tensorboard", "--logdir", str(logdir), "--port", str(port), group_dev=False)


@app.command("gguf-help")
def gguf_help() -> None:
    """Show llama.cpp GGUF conversion help."""
    try:
        uv_run(
            "python", "tools/llama_cpp/convert-hf-to-gguf.py", "--help", group_dev=False
        )
    except CommandError:
        typer.echo("[info] GGUF converter exited with a non-zero status", err=True)


def _coverage_file() -> Path:
    return CACHE_DIR / "coverage" / "coverage.sqlite"


@app.command("coverage-test")
def coverage_test() -> None:
    """Run targeted tests under coverage to collect data."""
    _ensure_cache_dirs()
    dest_cov = _coverage_file()
    dest_cov.parent.mkdir(parents=True, exist_ok=True)
    for path in dest_cov.parent.glob("coverage.sqlite*"):
        if path != dest_cov:
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
    env = os.environ.copy()
    env.update(
        {
            "HYPOTHESIS_DATABASE_DIRECTORY": str(CACHE_DIR / "hypothesis"),
            "HYPOTHESIS_STORAGE_DIRECTORY": str(CACHE_DIR / "hypothesis"),
            "HYPOTHESIS_SEED": "0",
            "PYTHONHASHSEED": "0",
            "COVERAGE_FILE": str(dest_cov),
        }
    )
    uv_run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "-n",
        "0",
        "tests/unit",
        "tests/property",
        env=env,
    )
    uv_run("coverage", "combine", env=env)
    for path in dest_cov.parent.glob("coverage.sqlite.*"):
        path.unlink(missing_ok=True)  # type: ignore[arg-type]


@app.command("coverage-report")
def coverage_report(
    verbose: bool = typer.Option(
        False, "--verbose", help="Print discovered coverage artifacts"
    ),
) -> None:
    """Generate coverage reports under .cache/coverage."""
    dest_cov = _coverage_file()
    if not any(dest_cov.parent.glob("coverage.sqlite*")):
        coverage_test()
    else:
        uv_run("coverage", "combine", env={"COVERAGE_FILE": str(dest_cov)})
    env = {"COVERAGE_FILE": str(dest_cov)}
    uv_run("coverage", "report", "-m", "--fail-under=0", env=env)
    uv_run(
        "coverage",
        "html",
        "-d",
        str(dest_cov.parent / "htmlcov"),
        "--fail-under=0",
        env=env,
    )
    uv_run(
        "coverage",
        "json",
        "-o",
        str(dest_cov.parent / "coverage.json"),
        "--fail-under=0",
        env=env,
    )
    uv_run(
        "coverage",
        "xml",
        "-o",
        str(dest_cov.parent / "coverage.xml"),
        "--fail-under=0",
        env=env,
    )
    if verbose:
        typer.echo("[coverage] artifacts:")
        for path in sorted(dest_cov.parent.iterdir()):
            typer.echo(f"  - {path.relative_to(ROOT)}")


@app.command("coverage-badge")
def coverage_badge() -> None:
    """Regenerate the SVG coverage badges."""
    cov_json = _coverage_file().parent / "coverage.json"
    if not cov_json.exists():
        coverage_report()
    uv_run("python", "tools/coverage_badges.py", str(cov_json), "docs/assets")


@mutation_app.command("reset")
def mutation_reset() -> None:
    """Remove the cached Cosmic Ray session."""
    session = _session_file()
    if session.exists():
        typer.echo(f"Removing {session}")
        session.unlink()


@mutation_app.command("summary")
def mutation_summary() -> None:
    """Show a summary of the previous Cosmic Ray run."""
    uv_run("python", "tools/mutation_summary.py", "--config", "pyproject.toml")


@mutation_app.command("init")
def mutation_init() -> None:
    """Initialize the Cosmic Ray session database if needed."""
    session = _session_file()
    session.parent.mkdir(parents=True, exist_ok=True)
    result = uv_run(
        "cosmic-ray",
        "init",
        "pyproject.toml",
        str(session),
        check=False,
    )
    if result.returncode != 0:
        typer.echo("[mutation] init skipped (reusing existing session)")
    else:
        typer.echo("[mutation] init complete")


@mutation_app.command("exec")
def mutation_exec() -> None:
    """Execute mutation tests with Cosmic Ray."""
    typer.echo("[mutation] starting exec")
    try:
        uv_run("cosmic-ray", "exec", "pyproject.toml", str(_session_file()))
    except CommandError as exc:
        typer.echo(f"[warning] Cosmic Ray returned non-zero status: {exc}", err=True)
        raise typer.Exit(1) from exc


@mutation_app.command("report")
def mutation_report() -> None:
    """Render a mutation testing report."""
    uv_run("python", "tools/mutation_report.py", "--config", "pyproject.toml")


@mutation_app.command("run")
def mutation_pipeline() -> None:
    """Run the full mutation testing pipeline."""
    mutation_reset()
    mutation_summary()
    mutation_init()
    mutation_exec()
    mutation_report()


@app.command("quality-ext")
def quality_ext() -> None:
    """Run quality gates followed by mutation testing."""
    quality()
    mutation_pipeline()


@lit_app.command("setup")
def lit_setup(
    python_version: str = typer.Option(
        "3.12", "--python-version", help="Python version for the isolated venv"
    ),
    recreate: bool = typer.Option(
        False, "--recreate", help="Recreate the LIT virtual environment"
    ),
) -> None:
    """Create a dedicated Python 3.12 environment for the LIT demo."""
    if recreate and LIT_VENV.exists():
        typer.echo(f"Removing {LIT_VENV}")
        shutil.rmtree(LIT_VENV, ignore_errors=True)
    uv("venv", "--python", python_version, str(LIT_VENV))
    uv(
        "pip",
        "install",
        "-p",
        str(_lit_python()),
        "-r",
        str(LIT_REQUIREMENTS),
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
    python_bin = _lit_python()
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
    uv_run(*args, python=str(python_bin), group_dev=False, no_project=True)


@lit_app.command("stop")
def lit_stop(
    port: int = typer.Option(5432, "--port", help="Port to terminate"),
    graceful: bool = typer.Option(
        True, "--graceful/--force", help="Attempt graceful shutdown"
    ),
) -> None:
    """Stop the LIT demo server bound to the given port."""
    python_bin = _lit_python()
    if not python_bin.exists():
        typer.echo("[info] LIT environment not found; nothing to stop")
        return
    args = ["python", "tools/port_kill.py", "--port", str(port)]
    if graceful:
        args.append("--graceful")
    uv_run(*args, python=str(python_bin), group_dev=False, no_project=True)


def main() -> None:  # pragma: no cover - Typer CLI entrypoint
    try:
        app()
    except CommandError as exc:
        raise typer.Exit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = [
#   "typer>=0.12.3",
# ]
# ///
"""Continuous integration workflows for ml_playground."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

import typer

from tools import task_utils as utils

app = typer.Typer(help="CI-oriented commands executed via uvx.", no_args_is_help=True)
mutation_app = typer.Typer(help="Mutation testing helpers")
app.add_typer(mutation_app, name="mutation")


CoverageFragment = Literal["unit", "property"]


def _coverage_fragment_path(fragment: CoverageFragment) -> Path:
    dest_cov = utils.coverage_file()
    return dest_cov.with_name(f"{dest_cov.name}.{fragment}")


def _coverage_run_env(coverage_file: Path) -> dict[str, str]:
    utils.ensure_cache_dirs("coverage", "hypothesis")
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HYPOTHESIS_DATABASE_DIRECTORY": str(utils.CACHE_DIR / "hypothesis"),
            "HYPOTHESIS_STORAGE_DIRECTORY": str(utils.CACHE_DIR / "hypothesis"),
            "HYPOTHESIS_SEED": "0",
            "PYTHONHASHSEED": "0",
            "COVERAGE_FILE": str(coverage_file),
        }
    )
    return env


def _coverage_file_env(coverage_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_file)
    return env


def _combine_fragments(
    dest_cov: Path, fragments: list[Path], *, ci_strict: bool
) -> None:
    if not fragments:
        return
    dest_cov.parent.mkdir(parents=True, exist_ok=True)
    if dest_cov.exists():
        utils.remove_path(dest_cov)
    env_combine = _coverage_file_env(dest_cov)
    result = utils.uv_run(
        "coverage",
        "combine",
        *(str(fragment) for fragment in fragments),
        env=env_combine,
        check=False,
    )
    if result.returncode != 0:
        message = "[coverage] failed to combine coverage fragments"
        if ci_strict:
            typer.echo(message, err=True)
            raise typer.Exit(result.returncode or 1)
        raise typer.Exit(result.returncode)
    for fragment in fragments:
        utils.remove_path(fragment)


def _run_coverage_fragment(
    fragment: CoverageFragment, args: Optional[List[str]] = None
) -> Path:
    fragment_path = _coverage_fragment_path(fragment)
    utils.remove_path(fragment_path)
    env = _coverage_run_env(fragment_path)
    targets = {
        "unit": ["tests/unit"],
        "property": ["tests/property"],
    }[fragment]
    extra = utils.forwarded_args(args)
    command = utils.pytest_command(targets + extra)
    try:
        n_index = command.index("-n")
        command[n_index + 1] = "0"
    except ValueError:
        pass
    utils.uv_run(
        "coverage",
        "run",
        f"--data-file={fragment_path}",
        "-m",
        *command,
        env=env,
    )
    shards = utils.coverage_fragments(fragment_path)
    if shards:
        _combine_fragments(fragment_path, shards, ci_strict=False)

    if not fragment_path.exists():
        typer.echo(
            f"[coverage] fragment generation produced no data for {fragment}",
            err=True,
        )
        raise typer.Exit(1)
    return fragment_path


@app.command()
def lint() -> None:
    """Run Ruff lint checks."""
    utils.uv_run("ruff", "check", ".")


@app.command("lint-check")
def lint_check() -> None:
    """Run Ruff in check-only mode (alias)."""
    lint()


def _pytest(targets: List[str]) -> None:
    utils.uv_run(*utils.pytest_command(targets))


@app.command()
def test(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run the full test suite."""
    _pytest(["tests", *utils.forwarded_args(args)])


@app.command()
def unit(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run unit tests."""
    _pytest(["tests/unit", *utils.forwarded_args(args)])


@app.command("property")
def property_tests(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run property-based tests."""
    _pytest(["tests/property", *utils.forwarded_args(args)])


@app.command()
def integration(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run integration tests."""
    _pytest(["-m", "integration", "--no-cov", *utils.forwarded_args(args)])


@app.command()
def e2e(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run end-to-end tests."""
    _pytest(["tests/e2e", *utils.forwarded_args(args)])


@app.command()
def acceptance(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pytest arguments", metavar="PYTEST_ARGS"
    ),
) -> None:
    """Run acceptance tests."""
    _pytest(["tests/acceptance", *utils.forwarded_args(args)])


@app.command("coverage-test")
def coverage_test() -> None:
    """Run targeted tests under coverage to collect data."""
    dest_cov = utils.coverage_file()
    utils.remove_path(dest_cov)
    fragments = [
        _run_coverage_fragment("unit"),
        _run_coverage_fragment("property"),
    ]
    _combine_fragments(dest_cov, fragments, ci_strict=False)


@app.command("coverage-report")
def coverage_report(
    fail_under: float = typer.Option(
        0.0,
        "--fail-under",
        help="Fail if total coverage is below this threshold.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print discovered coverage artifacts"
    ),
) -> None:
    """Generate coverage reports under .cache/coverage."""
    dest_cov = utils.coverage_file()
    ci_strict = os.environ.get("CI", "").lower() == "true"
    fragments = utils.coverage_fragments(dest_cov)

    if fragments:
        _combine_fragments(dest_cov, fragments, ci_strict=ci_strict)

    if not dest_cov.exists():
        typer.echo(
            "[coverage] coverage data file is missing; run `uvx --from . ci-tasks coverage-test` or provide coverage fragments",
            err=True,
        )
        raise typer.Exit(1)

    if ci_strict and dest_cov.stat().st_size == 0:
        typer.echo("[coverage] coverage data file is empty", err=True)
        raise typer.Exit(1)

    env = _coverage_file_env(dest_cov)
    fail_arg = ["--fail-under", f"{fail_under:.2f}"]
    coverage_dir = dest_cov.parent
    commands: list[tuple[str, list[str]]] = [
        ("report", ["-m", *fail_arg]),
        ("html", ["-d", str(coverage_dir / "htmlcov")]),
        ("json", ["-o", str(coverage_dir / "coverage.json")]),
        ("xml", ["-o", str(coverage_dir / "coverage.xml")]),
    ]

    for subcommand, args in commands:
        utils.uv_run("coverage", subcommand, *args, env=env)

    if verbose:
        typer.echo("[coverage] artifacts:")
        for path in sorted(dest_cov.parent.iterdir()):
            typer.echo(f"  - {path.relative_to(utils.ROOT)}")


@app.command("coverage-badge")
def coverage_badge() -> None:
    """Regenerate the SVG coverage badges."""
    json_path = utils.coverage_file().parent / "coverage.json"
    if not json_path.exists():
        coverage_report()
    utils.uv_run("python", "tools/coverage_badges.py", str(json_path), "docs/assets")


@app.command()
def quality() -> None:
    """Run the full pre-commit quality gate."""
    utils.uv_run(
        "pre-commit", "run", "--config", str(utils.PRE_COMMIT_CONFIG), "--all-files"
    )


@app.command("quality-fast")
def quality_fast() -> None:
    """Run lint/format focused pre-commit hooks."""
    utils.uv_run(
        "pre-commit",
        "run",
        "--config",
        str(utils.PRE_COMMIT_CONFIG),
        "--all-files",
        "ruff",
    )
    utils.uv_run(
        "pre-commit",
        "run",
        "--config",
        str(utils.PRE_COMMIT_CONFIG),
        "--all-files",
        "ruff-format",
    )
    utils.uv_run(
        "pre-commit",
        "run",
        "--config",
        str(utils.PRE_COMMIT_CONFIG),
        "--all-files",
        "mdformat",
    )


@app.command("quality-ext")
def quality_ext() -> None:
    """Run quality gates followed by mutation testing."""
    quality()
    mutation_run()


@mutation_app.command("reset")
def mutation_reset() -> None:
    """Remove the cached Cosmic Ray session."""
    session = utils.cosmic_ray_session_file()
    if session.exists():
        typer.echo(f"Removing {session}")
        utils.remove_path(session)


@mutation_app.command("summary")
def mutation_summary() -> None:
    """Show a summary of the previous Cosmic Ray run."""
    utils.uv_run("python", "tools/mutation_summary.py", "--config", "pyproject.toml")


@mutation_app.command("init")
def mutation_init() -> None:
    """Initialize the Cosmic Ray session database if needed."""
    session = utils.cosmic_ray_session_file()
    session.parent.mkdir(parents=True, exist_ok=True)
    result = utils.uv_run(
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
        utils.uv_run(
            "cosmic-ray", "exec", "pyproject.toml", str(utils.cosmic_ray_session_file())
        )
    except utils.CommandError as exc:
        typer.echo(f"[warning] Cosmic Ray returned non-zero status: {exc}", err=True)
        raise typer.Exit(1) from exc


@mutation_app.command("report")
def mutation_report() -> None:
    """Render a mutation testing report."""
    utils.uv_run("python", "tools/mutation_report.py", "--config", "pyproject.toml")


@mutation_app.command("run")
def mutation_run() -> None:
    """Run the full mutation testing pipeline."""
    mutation_reset()
    mutation_summary()
    mutation_init()
    mutation_exec()
    mutation_report()


def main() -> None:  # pragma: no cover
    try:
        app()
    except utils.CommandError as exc:  # pragma: no cover
        raise typer.Exit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()

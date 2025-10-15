#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = [
#   "typer>=0.12.3",
# ]
# ///
"""Continuous integration workflows for ml_playground."""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import List, Literal, Optional

import typer

from tools import task_utils as utils

app = typer.Typer(
    help="CI-oriented commands executed via uv run.", no_args_is_help=True
)
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
    dest_cov: Path,
    fragments: list[Path],
    *,
    ci_strict: bool,
    cleanup: bool = True,
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
    if cleanup:
        for fragment in fragments:
            utils.remove_path(fragment)


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
    utils.ensure_cache_dirs("coverage", "hypothesis")
    dest_cov = utils.coverage_file()
    dest_cov.parent.mkdir(parents=True, exist_ok=True)
    utils.remove_path(dest_cov)
    for fragment in utils.coverage_fragments(dest_cov):
        utils.remove_path(fragment)

    env = os.environ.copy()
    env.update(
        {
            "HYPOTHESIS_DATABASE_DIRECTORY": str(utils.CACHE_DIR / "hypothesis"),
            "HYPOTHESIS_STORAGE_DIRECTORY": str(utils.CACHE_DIR / "hypothesis"),
            "HYPOTHESIS_SEED": "0",
            "PYTHONHASHSEED": "0",
            "COVERAGE_FILE": str(dest_cov),
        }
    )
    utils.uv_run(
        "coverage",
        "run",
        f"--data-file={dest_cov}",
        "-m",
        "pytest",
        "-n",
        "0",
        "tests/unit",
        "tests/property",
        env=env,
    )
    # Normalize to a single monolithic file so the report stage can rely on it
    env_combine = _coverage_file_env(dest_cov)
    utils.uv_run("coverage", "combine", env=env_combine, check=False)


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
    if not dest_cov.exists():
        typer.echo(
            "[coverage] missing coverage data file. Ensure the prior 'coverage-test' step ran and wrote '"
            + str(dest_cov)
            + "'",
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


@app.command("coverage-threshold")
def coverage_threshold(
    line_threshold: float = typer.Option(
        0.0,
        "--line-threshold",
        help="Fail if total line coverage is below this percentage.",
    ),
    branch_threshold: float = typer.Option(
        0.0,
        "--branch-threshold",
        help="Fail if total branch coverage is below this percentage.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Print computed coverage totals."
    ),
) -> None:
    """Fail when coverage metrics drop below configured thresholds."""
    dest_cov = utils.coverage_file()
    if not dest_cov.exists():
        typer.echo(
            "[coverage] missing coverage data file. Run 'uv run ci-tasks coverage-test' first.",
            err=True,
        )
        raise typer.Exit(1)

    env = _coverage_file_env(dest_cov)
    json_path = dest_cov.parent / "coverage.json"
    utils.uv_run("coverage", "json", "-o", str(json_path), env=env)

    try:
        totals = json.loads(json_path.read_text(encoding="utf-8"))["totals"]
    except (json.JSONDecodeError, KeyError) as exc:  # pragma: no cover - defensive
        typer.echo(f"[coverage] failed to parse branch coverage data: {exc}", err=True)
        raise typer.Exit(1)

    num_branches = totals.get("num_branches", 0)
    covered_branches = totals.get("covered_branches", 0)
    covered_lines = totals.get("covered_lines", 0)
    num_statements = totals.get("num_statements", 0)

    messages: list[str] = []

    line_pct = (covered_lines / num_statements) * 100 if num_statements else 0.0
    if verbose:
        typer.echo(
            f"[coverage] totals: lines={line_pct:.2f}% branches="
            f"{(covered_branches / num_branches * 100) if num_branches else float('nan'):.2f}%"
        )

    if line_threshold > 0 and num_statements == 0:
        messages.append("line coverage totals missing from coverage.json")
    elif line_threshold > 0 and line_pct < line_threshold:
        messages.append(
            f"Line coverage {line_pct:.2f}% < {line_threshold:.2f}%. Run 'uv run ci-tasks coverage-test'."
        )

    if branch_threshold > 0:
        if num_branches == 0:
            messages.append("Branch coverage data missing from coverage.json")
        else:
            branch_pct = (covered_branches / num_branches) * 100
            if branch_pct < branch_threshold:
                messages.append(
                    f"Branch coverage {branch_pct:.2f}% < {branch_threshold:.2f}%."
                )

    if messages:
        for message in messages:
            typer.echo(f"[coverage] {message}", err=True)
        raise typer.Exit(1)


@app.command("coverage-badge")
def coverage_badge() -> None:
    """Regenerate the SVG coverage badges."""
    json_path = utils.coverage_file().parent / "coverage.json"
    if not json_path.exists():
        coverage_report()
    utils.uv_run("python", "tools/coverage_badges.py", str(json_path), "docs/assets")


@app.command()
def quality(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pre-commit arguments", metavar="PRECOMMIT_ARGS"
    ),
) -> None:
    """Run the full pre-commit quality gate."""
    utils.uv_run(
        "pre-commit",
        "run",
        "--config",
        str(utils.PRE_COMMIT_CONFIG),
        "--all-files",
        *utils.forwarded_args(args),
    )
    integration()
    acceptance()
    e2e()


@app.command("quality-fast")
def quality_fast(
    args: Optional[List[str]] = typer.Argument(
        None, help="Additional pre-commit arguments", metavar="PRECOMMIT_ARGS"
    ),
) -> None:
    """Run lint/format focused pre-commit hooks."""
    utils.uv_run(
        "pre-commit",
        "run",
        "--config",
        str(utils.PRE_COMMIT_CONFIG),
        "--all-files",
        "ruff",
        *utils.forwarded_args(args),
    )
    utils.uv_run(
        "pre-commit",
        "run",
        "--config",
        str(utils.PRE_COMMIT_CONFIG),
        "--all-files",
        "ruff-format",
        *utils.forwarded_args(args),
    )
    utils.uv_run(
        "pre-commit",
        "run",
        "--config",
        str(utils.PRE_COMMIT_CONFIG),
        "--all-files",
        "mdformat",
        *utils.forwarded_args(args),
    )


@app.command("quality-ext")
def quality_ext() -> None:
    """Run quality gates followed by mutation testing."""
    quality()
    mutation_run()


@app.command("quality-ci-local")
def quality_ci_local(
    bind_caches: bool = typer.Option(
        True,
        "--bind-caches/--no-bind-caches",
        help="Bind local caches and the project virtualenv into the act container.",
    ),
    args: Optional[List[str]] = typer.Argument(
        None,
        help="Additional arguments forwarded to act.",
        metavar="ACT_ARGS",
    ),
) -> None:
    """Run the GitHub quality workflow locally using act."""
    utils.ensure_cache_dirs("uv", "pre-commit", "ruff")
    (utils.ROOT / ".venv").mkdir(parents=True, exist_ok=True)

    command: List[str] = [
        "act",
        "--container-architecture",
        "linux/amd64",
        "-P",
        "ubuntu-latest=catthehacker/ubuntu:act-latest",
        "-W",
        ".github/workflows/quality.yml",
        "--job",
        "quality",
    ]

    if bind_caches:
        binds: list[tuple[Path, str]] = [
            (utils.CACHE_DIR / "uv", "/root/.cache/uv"),
            (utils.CACHE_DIR / "pre-commit", "/root/.cache/pre-commit"),
            (utils.CACHE_DIR / "ruff", "/root/.cache/ruff"),
            (utils.ROOT / ".venv", "/root/project/.venv"),
        ]
        for host_path, container_path in binds:
            host_path.mkdir(parents=True, exist_ok=True)
            command.extend(["--bind", f"{host_path}:{container_path}"])

    command.extend(utils.forwarded_args(args))

    result = subprocess.run(command, cwd=utils.ROOT)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


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

"""Shared utilities for uv-backed task CLIs."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import typer


def _discover_root() -> Path:
    """Locate the repository root by walking up to the first `pyproject.toml`."""

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


class CommandError(RuntimeError):
    """Raised when an invoked subprocess fails."""


def _echo_command(command: List[str]) -> None:
    formatted = " ".join(shlex.quote(arg) for arg in command)
    typer.echo(f"$ {formatted}")


def _run(
    command: List[str], *, env: Optional[dict[str, str]] = None, check: bool = True
) -> subprocess.CompletedProcess:
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
    python: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    no_project: bool = False,
) -> subprocess.CompletedProcess:
    command: List[str] = ["uv", "run"]
    if no_project:
        command.append("--no-project")
    else:
        command.extend(["--project", str(ROOT)])
    if python:
        command.extend(["--python", python])
    command.extend(args)
    return _run(command, env=env, check=check)


def ensure_cache_dirs(*subdirs: str) -> None:
    for subdir in subdirs:
        (CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def forwarded_args(args: Optional[List[str]]) -> List[str]:
    return list(args or [])


def pytest_command(extra: Optional[List[str]] = None) -> List[str]:
    return ["pytest", *PYTEST_BASE, *(extra or [])]


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)  # type: ignore[arg-type]


def coverage_file() -> Path:
    return CACHE_DIR / "coverage" / "coverage.sqlite"


def coverage_fragments(dest: Path) -> list[Path]:
    return [p for p in dest.parent.glob("coverage.sqlite.*") if p.name != dest.name]


def cosmic_ray_session_file() -> Path:
    return CACHE_DIR / "cosmic-ray" / "session.sqlite"


def lit_python() -> Path:
    if os.name == "nt":
        return LIT_VENV / "Scripts" / "python.exe"
    return LIT_VENV / "bin" / "python"

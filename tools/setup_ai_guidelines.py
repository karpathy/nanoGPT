#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict
import typer

app = typer.Typer(add_completion=False)

# ---- Constants ----
README_NAME = "Readme.md"
PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT_DIR / "tools"
BASE_DIR = PROJECT_DIR / ".dev-guidelines"

# Tool configuration:
#   maps tool name -> relative path (from project dir) for the primary symlink to Readme.md
TOOL_MAP: Dict[str, str] = {
    "copilot": ".github/copilot-instructions.md",
    "aiassistant": ".aiassistant/rules/00-Readme.md",
    "junie": ".junie/guidelines.md",
    "kiro": ".kiro/steering/product.md",
    "windsurf": ".windsurf/rules/rule.md",
    "cursor": ".cursor/rules/00-readme.mdc",
}


# ---- Helpers ----
def info(msg: str) -> None:
    typer.echo(msg)


def warn(msg: str) -> None:
    typer.echo(typer.style(f"WARNING: {msg}", fg=typer.colors.YELLOW))


def err(msg: str) -> None:
    typer.echo(typer.style(f"ERROR: {msg}", fg=typer.colors.RED))


def ensure_dir(path: Path, dry_run: bool) -> None:
    """Ensure a path exists, creating it if necessary.

    Automatically detects if the path is a file or directory based on whether it has a file extension.

    Args:
        path: The path to ensure.
        dry_run: If True, only print the action without executing.
    """

    if not path.exists():
        if dry_run:
            action = "touch" if path.is_file() else "mkdir -p"
            info(f"[dry-run] {action} {path}")
        else:
            if path.is_file():
                path.touch()
                info(f"create {path} (empty)")
            else:
                path.mkdir(parents=True, exist_ok=True)


def ensure_base_and_empty_readme(dry_run: bool) -> Path:
    """Ensure the base directory and empty README file exist.

    Args:
        dry_run: If True, only print actions without executing.

    Returns:
        Path to the README file.
    """
    ensure_dir(BASE_DIR, dry_run)
    readme = BASE_DIR / README_NAME
    ensure_dir(readme, dry_run)
    return readme


def create_or_update_symlink(link_path: Path, target_path: Path, dry_run: bool) -> None:
    """
    Create/refresh a relative symlink. If a non-symlink file/dir exists, skip.
    """
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            try:
                current = (link_path.parent / link_path.readlink()).resolve()
            except OSError:
                current = None
            if current == target_path.resolve():
                info(f"ok     {link_path} -> {target_path}")
                return
            if dry_run:
                info(f"[dry-run] rm {link_path}")
            else:
                link_path.unlink()
        else:
            warn(f"skip   {link_path} exists and is not a symlink")
            return

    ensure_dir(link_path.parent, dry_run)
    if dry_run:
        info(f"[dry-run] ln -s {target_path} {link_path}")
    else:
        rel = os.path.relpath(target_path, start=link_path.parent)
        link_path.symlink_to(rel)
        info(f"link   {link_path} -> {target_path}")


def clean_broken_symlinks_pointing_into_base(scan_dir: Path, dry_run: bool) -> None:
    """
    Remove any symlink under scan_dir that points into BASE_DIR but whose target is missing.
    """
    if not scan_dir.exists():
        return
    for path in scan_dir.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            target = (path.parent / path.readlink()).resolve()
        except OSError:
            target = None
        if (
            target
            and str(target).startswith(str(BASE_DIR.resolve()))
            and not target.exists()
        ):
            if dry_run:
                info(f"[dry-run] rm broken symlink {path} (-> {target})")
            else:
                path.unlink()
                info(f"clean  removed broken symlink {path}")


def is_ignored_by_git(tool_dir: Path) -> bool:
    """Check if a directory is ignored by Git.

    Args:
        tool_dir: The directory path to check.

    Returns:
        True if the directory is ignored by Git, False otherwise.
    """
    if not (PROJECT_DIR / ".git").exists():
        return False

    try:
        result = subprocess.run(
            ["git", "check-ignore", str(tool_dir)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def ask_gitignore_for_dir(tool_dir: Path, dry_run: bool) -> None:
    """Ask to add tool_dir (as a trailing-slash path) to .gitignore if not present.

    Args:
        tool_dir: The directory to potentially add to .gitignore.
        dry_run: If True, only print actions without executing.
    """
    gitignore_path = PROJECT_DIR / ".gitignore"
    relative_path = os.path.relpath(tool_dir, start=PROJECT_DIR).rstrip("/") + "/"

    if is_ignored_by_git(tool_dir):
        info(f"git    '{relative_path}' already ignored by git")
        return

    if typer.confirm(f"Add '{relative_path}' to .gitignore?", default=False):
        if dry_run:
            info(f"[dry-run] append '{relative_path}' to {gitignore_path}")
        else:
            with gitignore_path.open("a", encoding="utf-8") as f:
                if gitignore_path.exists() and gitignore_path.stat().st_size > 0:
                    f.write("\n")
                f.write(relative_path + "\n")
            info(f"git    added '{relative_path}' to .gitignore")
    else:
        info("git    skipped .gitignore update")


# ---- Command ----
@app.command()
def setup(
    tool: str = typer.Argument(
        ..., help=f"One of: {', '.join(sorted(TOOL_MAP.keys()))}"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run", help="Show actions without making changes."
    ),
):
    """
    Create symlinks for all files in the base directory to the tool's directory.
    Paths are resolved from the project (tools) directory.
    """
    tool = tool.lower()
    if tool not in TOOL_MAP:
        err(f"Unknown tool '{tool}'. Supported: {', '.join(sorted(TOOL_MAP))}")
        raise typer.Exit(code=1)

    tool_path = Path(TOOL_MAP[tool])

    # 1) Ensure base + empty README
    readme = ensure_base_and_empty_readme(dry_run)

    # 2) Ensure tool's directory exists (under PROJECT_DIR)
    tool_dir = (PROJECT_DIR / tool_path.parent).resolve()
    ensure_dir(tool_dir, dry_run)

    # 3) Create primary symlink from tool map path to README
    primary_path = (PROJECT_DIR / tool_path).resolve()
    ensure_dir(primary_path.parent, dry_run)
    create_or_update_symlink(primary_path, readme, dry_run=dry_run)

    # 4) Create symlinks for all files in BASE_DIR (using list comprehension)
    links = [
        ((tool_dir / file_path.name).resolve(), file_path)
        for file_path in BASE_DIR.iterdir()
        if file_path.is_file() and file_path != readme
    ]

    for link_path, target in links:
        ensure_dir(link_path.parent, dry_run)
        create_or_update_symlink(link_path, target, dry_run=dry_run)

    # 5) Clean broken symlinks pointing into BASE_DIR within tool's directory
    clean_broken_symlinks_pointing_into_base(tool_dir, dry_run)

    # 6) Ask about .gitignore for the tool's directory
    ask_gitignore_for_dir(tool_dir, dry_run)

    info("done.")


if __name__ == "__main__":
    app()

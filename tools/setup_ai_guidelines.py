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
    # "codex": "Agent.md" - for codex we would need to merge or hope to
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
    # If it already exists, nothing to do
    try:
        if path.exists():
            return
    except OSError:
        # On some platforms calling exists() can raise for problematic paths; fall through to create logic
        pass

    # Heuristic: treat as file if it has a suffix (e.g., "Readme.md"); directories like ".github" have no suffix
    is_file_like = path.suffix != ""

    if dry_run:
        action = "touch" if is_file_like else "mkdir -p"
        info(f"[dry-run] {action} {path}")
        return

    if is_file_like:
        # Ensure parent directory first, then create an empty file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
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


def _windows_create_junction(link_path: Path, target_path: Path) -> None:
    """Create a directory junction on Windows using mklink /J.

    link_path must not already exist.
    """
    # Use absolute paths for mklink
    cmd = [
        "cmd",
        "/c",
        "mklink",
        "/J",
        str(link_path),
        str(target_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to create junction {link_path} -> {target_path}: {result.stderr.strip()}"
        )


def create_or_update_link(link_path: Path, target_path: Path, dry_run: bool) -> None:
    """
    Create/refresh a link from link_path to target_path.

    Policy:
    - Windows: files -> hardlinks; directories -> junctions.
    - Unix-like: symlinks for both files and directories (relative path when feasible).
    - Strict: if an incompatible existing path is present, raise instead of silently copying.
    """
    link_exists = False
    link_is_symlink = False
    try:
        link_exists = link_path.exists()
        link_is_symlink = link_path.is_symlink()
    except OSError:
        # On Windows, certain paths can raise when probing; assume non-existent
        link_exists = False
        link_is_symlink = False

    is_windows = os.name == "nt"
    try:
        target_is_dir = target_path.is_dir()
    except OSError:
        target_is_dir = False

    # If link_path and target_path are effectively the same, skip
    try:
        same = os.path.samefile(link_path, target_path)
    except OSError:
        same = link_path.resolve() == target_path.resolve()
    if same:
        info(f"ok     {link_path} == {target_path} (same path)")
        return

    # If something exists at link_path, handle per OS policy
    if link_exists or link_is_symlink:
        if dry_run:
            info(f"[dry-run] rm {link_path}")
            # Continue to create link below in dry-run mode
        else:
            # Remove existing link/symlink/file; do not recursively delete directories
            try:
                if link_path.is_symlink() or link_path.is_file():
                    link_path.unlink()
                elif link_path.is_dir():
                    # Only allow removing empty dir to avoid destructive behavior
                    if any(link_path.iterdir()):
                        raise RuntimeError(
                            f"Cannot replace non-empty directory at {link_path}. Remove it first."
                        )
                    link_path.rmdir()
                else:
                    # Unknown type; attempt unlink
                    link_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError as e:
                raise RuntimeError(
                    f"failed to remove existing path {link_path}: {e}"
                ) from e

    ensure_dir(link_path.parent, dry_run)
    if dry_run:
        if is_windows and not target_is_dir:
            info(f"[dry-run] hardlink {link_path} -> {target_path}")
        elif is_windows and target_is_dir:
            info(f"[dry-run] junction {link_path} -> {target_path}")
        else:
            info(f"[dry-run] ln -s {target_path} {link_path}")
        return

    # Create per-OS
    if is_windows:
        if target_is_dir:
            # Directory junction
            try:
                _windows_create_junction(link_path, target_path)
                info(f"link   {link_path} => {target_path} (junction)")
            except RuntimeError:
                raise
        else:
            # File hardlink (requires same volume)
            try:
                os.link(target_path, link_path)
                info(f"link   {link_path} == {target_path} (hardlink)")
            except OSError as e:
                raise RuntimeError(
                    f"failed to create hardlink {link_path} -> {target_path}: {e}. "
                    "Ensure both paths are on the same volume."
                ) from e
    else:
        # Unix-like: symlink for both files and directories (use relative path)
        rel = os.path.relpath(target_path, start=link_path.parent)
        try:
            if target_is_dir:
                link_path.symlink_to(rel, target_is_directory=True)
            else:
                link_path.symlink_to(rel)
            info(f"link   {link_path} -> {target_path}")
        except OSError as e:
            err(f"failed to create symlink {link_path} -> {target_path}: {e}")


def mirror_tree(
    src_dir: Path, dest_dir: Path, exclude: set[Path] | None, dry_run: bool
) -> None:
    """Mirror src_dir into dest_dir by linking each top-level entry.

    - Windows: files -> hardlinks, directories -> junctions.
    - Unix: symlinks for both.
    - Exclude: set of absolute paths under src_dir to skip.

    Note: We do NOT recurse into directories. A directory link replaces recursion and avoids
    creating links inside linked directories (which could mutate the source via junctions).
    """
    if not src_dir.exists():
        return
    for entry in src_dir.iterdir():
        target = entry.resolve()
        if exclude and target in exclude:
            continue
        dest_path = (dest_dir / entry.name).resolve()
        create_or_update_link(dest_path, target, dry_run)


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

    # 3) Create primary link from tool map path to README
    primary_path = (PROJECT_DIR / tool_path).resolve()
    ensure_dir(primary_path.parent, dry_run)
    create_or_update_link(primary_path, readme, dry_run=dry_run)

    # 4) Mirror entire BASE_DIR contents into tool_dir (recursive)
    # Exclude the README file to avoid duplication, since it already has a primary mapping above
    exclude: set[Path] = {readme.resolve()}
    mirror_tree(BASE_DIR.resolve(), tool_dir, exclude, dry_run)

    # 5) Clean broken symlinks pointing into BASE_DIR within tool's directory
    clean_broken_symlinks_pointing_into_base(tool_dir, dry_run)

    # 6) Ask about .gitignore for the tool's directory
    ask_gitignore_for_dir(tool_dir, dry_run)

    info("done.")


if __name__ == "__main__":
    app()

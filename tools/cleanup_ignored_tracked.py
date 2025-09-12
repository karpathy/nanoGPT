#!/usr/bin/env python3
"""
cleanup_ignored_tracked.py

Interactive tool (Typer-based) to manage files that are tracked in Git but
are ignored by your current .gitignore rules.

What this tool does:
  - Detects tracked files that match ignore patterns.
  - Lets you choose between a Dry Run (show command), Actually Run (REWRITE HISTORY to purge), or Cancel.

Safety and history:
  - The "Actually Run" option uses `git filter-repo --invert-paths` with the matched paths to REWRITE HISTORY and purge them from all commits.
  - This is destructive to commit hashes. You will need to force-push and coordinate with collaborators.
  - A dry run prints the exact command that would be executed. Cancel makes no changes.

Usage:
  - Simply run: `python tools/cleanup_ignored_tracked.py`
  - Follow the on-screen prompt to select an option.

Notes:
  - This script shells out to Git and requires being run inside a Git repo.
  - It respects .gitignore and other standard ignore files via `git check-ignore`.
"""

from __future__ import annotations

import os
import tempfile
import shlex
import subprocess
import sys
from typing import Iterable, List, Tuple

import typer
import click


def run(
    cmd: List[str], cwd: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def git_root(start: str = ".") -> str:
    try:
        out = run(["git", "rev-parse", "--show-toplevel"], cwd=start)
        return out.stdout.strip()
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error: not inside a Git repository.\n{e.stderr}", err=True)
        sys.exit(2)


def get_tracked_files(repo: str) -> List[str]:
    # List all tracked files
    out = run(["git", "ls-files"], cwd=repo)
    files = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return files


def filter_ignored(repo: str, paths: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Return a list of tuples (path, rule) for paths that are ignored by Git according to .gitignore.
    Uses `git check-ignore -v` to include the matching rule/source for transparency.
    """
    # git check-ignore -v --stdin prints lines like:
    # <source>:<lineno>:<pattern>\t<path>
    proc = subprocess.Popen(
        ["git", "check-ignore", "-v", "--stdin"],
        cwd=repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None

    for p in paths:
        proc.stdin.write(p + "\n")
    proc.stdin.close()

    results: List[Tuple[str, str]] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        # Expect format: source:lineno:pattern\tpath
        if "\t" in line:
            meta, path = line.split("\t", 1)
            results.append((path, meta))
        else:
            # Fallback: if not parsable, just include the line
            results.append((line, ""))

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    ret = proc.wait()
    if ret not in (0, 1):  # 1 may mean no matches, but we handled output above
        print("git check-ignore failed:", file=sys.stderr)
        print(stderr, file=sys.stderr)
        sys.exit(3)

    return results


def _check_filter_repo_available(repo: str) -> bool:
    """Return True if `git filter-repo` appears available on PATH.

    We consider the tool available if invoking `git filter-repo --help` does not
    raise FileNotFoundError. A non-zero exit code is fine (usage errors, etc.).
    """
    try:
        run(["git", "filter-repo", "--help"], cwd=repo, check=False)
        return True
    except FileNotFoundError:
        return False


def find_tracked_and_ignored(
    repo: str, limit_to_path: str | None = None
) -> List[Tuple[str, str]]:
    tracked = get_tracked_files(repo)
    if limit_to_path:
        limit_to_path = os.path.normpath(limit_to_path)
        tracked = [
            p
            for p in tracked
            if p == limit_to_path or p.startswith(limit_to_path.rstrip(os.sep) + os.sep)
        ]
    return filter_ignored(repo, tracked)


def print_list(items: List[Tuple[str, str]]) -> None:
    if not items:
        typer.echo("No tracked files are ignored by Git. ✅")
        return
    lines = [
        "Tracked files currently ignored by Git (source:line:pattern -> path):",
        *[f"  {meta} -> {path}" for path, meta in items],
        f"\nTotal: {len(items)} file(s)",
    ]
    typer.echo("\n".join(lines))


def rewrite_history(repo: str, items: List[Tuple[str, str]], dry_run: bool) -> int:
    if not items:
        print("Nothing to rewrite. ")
        return 0

    # Prepare a temporary file with all paths (one per line)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tmp_path = tf.name
            for path, _meta in items:
                tf.write(path + "\n")
    except Exception as e:
        typer.echo(f"Error preparing temporary paths file: {e}", err=True)
        return 1

    cmd = ["git", "filter-repo", "--invert-paths", "--paths-from-file", tmp_path]

    if dry_run:
        typer.echo(
            f"Planned to purge {len(items)} path(s) from history (see list above)."
        )
        typer.echo(f"\nDRY-RUN: would run: {shlex.join(cmd)}")
        # Clean up temp file after preview
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return 0

    if not _check_filter_repo_available(repo):
        typer.echo(
            "Error: 'git filter-repo' is not installed or not on PATH.", err=True
        )
        typer.echo(
            "Install: https://github.com/newren/git-filter-repo and re-run.", err=True
        )
        return 2

    # Final confirmation
    typer.echo(
        f"You are about to REWRITE HISTORY and purge {len(items)} path(s) from all commits."
    )
    typer.echo("This will change commit hashes. You will likely need to force-push.")
    proceed = typer.confirm("Proceed with history rewrite?", default=False)
    if not proceed:
        typer.echo("Canceled. No changes made.")
        return 0

    try:
        out = run(cmd, cwd=repo, check=True)
        # Show a concise echo of what happened
        sys.stdout.write(out.stdout)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error rewriting history: {e.stderr}", err=True)
        return 1
    finally:
        # Always clean up temp file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    typer.echo("\nHistory rewritten. Remember to force-push the changes.")
    return 0


def explain_and_prompt(items: List[Tuple[str, str]]) -> str:
    typer.echo(
        (
            "\nThis tool will:\n"
            "  • Find files that are tracked in Git but currently ignored by .gitignore.\n"
            "  • Optionally purge them from history using 'git filter-repo --invert-paths'.\n\n"
            "WARNING: The 'Run' option will REWRITE HISTORY using git filter-repo.\n"
            "         This changes commit hashes and requires a force-push.\n"
            "         Git will keep backups under refs/original/ for recovery.\n"
        )
    )
    print_list(items)
    typer.echo(
        "\nChoose an option:\n"
        "  [d] Dry run (show the git filter-repo command)\n"
        "  [r] Run now (rewrite history with git filter-repo)\n"
        "  [c] Cancel"
    )
    choice = typer.prompt(
        "Enter choice",
        default="c",
        type=click.Choice(["d", "r", "c"], case_sensitive=False),
    )
    return choice.lower()


def main() -> int:
    repo = git_root()
    # For now we operate on the entire repo; add scoped path here if needed.
    items = find_tracked_and_ignored(repo, limit_to_path=None)

    if not items:
        typer.echo("No tracked files are ignored by Git. ✅")
        return 0

    choice = explain_and_prompt(items)
    if choice == "d":
        return rewrite_history(repo, items, dry_run=True)
    if choice == "r":
        return rewrite_history(repo, items, dry_run=False)
    typer.echo("Canceled. No changes made.")
    return 0


if __name__ == "__main__":
    # Run Typer app entry (no commands/args needed).
    sys.exit(typer.run(main))

#!/usr/bin/env python3
"""
cleanup_ignored_tracked.py

Interactive tool (Typer-based) to manage files that are tracked in Git but
are ignored by your current .gitignore rules.

What this tool does:
  - Detects tracked files that match ignore patterns.
  - Lets you choose between a Dry Run (show commands), Actually Run (untrack), or Cancel.

Safety and history:
  - The "Actually Run" option uses `git rm --cached` to untrack files but keeps them on disk.
  - It does NOT rewrite history. Past commits remain unchanged.
  - If you need to purge files from history, use a separate history-rewrite tool (not included here).

Usage:
  - Simply run: `python tools/cleanup_ignored_tracked.py`
  - Follow the on-screen prompt to select an option.

Notes:
  - This script shells out to Git and requires being run inside a Git repo.
  - It respects .gitignore and other standard ignore files via `git check-ignore`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import Iterable, List, Tuple

import typer


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
        print("Error: not inside a Git repository.", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
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
        print("No tracked files are ignored by Git. ✅")
        return

    print("Tracked files currently ignored by Git (source:line:pattern -> path):")
    for path, meta in items:
        print(f"  {meta} -> {path}")
    print(f"\nTotal: {len(items)} file(s)")


def delete_items(
    repo: str, items: List[Tuple[str, str]], dry_run: bool, wipe: bool
) -> int:
    if not items:
        print("Nothing to delete. ✅")
        return 0

    # Choose command template
    base_cmd = ["git", "rm"]
    if not wipe:
        base_cmd.append("--cached")

    exit_code = 0
    for path, meta in items:
        cmd = base_cmd + ["--", path]
        if dry_run:
            print(f"DRY-RUN: would run: {shlex.join(cmd)}   # ignored by {meta}")
            continue
        try:
            out = run(cmd, cwd=repo, check=True)
            # Show a concise echo of what happened
            sys.stdout.write(out.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error removing {path}: {e.stderr}", file=sys.stderr)
            exit_code = 1

    if dry_run:
        print("\nDry run complete. Re-run to apply.")
    else:
        print("\nRemoval complete. Remember to commit the changes.")
    return exit_code


def explain_and_prompt(items: List[Tuple[str, str]]) -> str:
    print()
    print("This tool will:" )
    print("  • Find files that are tracked in Git but currently ignored by .gitignore.")
    print("  • Optionally untrack them using 'git rm --cached' (keeps files on disk).")
    print()
    print("It will NOT rewrite history. Past commits remain unchanged.")
    print("To purge from history, use a separate history-rewrite tool (e.g., git filter-repo).")
    print()
    print_list(items)
    print()
    print("Choose an option:")
    print("  [d] Dry run (show what would be done)")
    print("  [r] Run now (untrack with 'git rm --cached')")
    print("  [c] Cancel")
    choice = typer.prompt("Enter choice", default="c").strip().lower()
    return choice


def main() -> int:
    repo = git_root()
    # For now we operate on the entire repo; add scoped path here if needed.
    items = find_tracked_and_ignored(repo, limit_to_path=None)

    if not items:
        print("No tracked files are ignored by Git. ✅")
        return 0

    choice = explain_and_prompt(items)
    if choice in ("d", "dry", "dry-run", "dryrun"):
        return delete_items(repo, items, dry_run=True, wipe=False)
    if choice in ("r", "run", "apply", "go"):
        return delete_items(repo, items, dry_run=False, wipe=False)
    print("Canceled. No changes made.")
    return 0


if __name__ == "__main__":
    # Run Typer app entry (no commands/args needed).
    sys.exit(typer.run(main))

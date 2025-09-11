#!/usr/bin/env python3
"""
cleanup_ignored_tracked.py

Utility to manage tracked files that match .gitignore rules.

Modes:
  - check:  Print tracked files that are ignored by Git
  - delete: Remove those files from the repo (dry-run by default)

Delete mode behavior:
  - By default, performs a dry run and shows the `git rm` commands it would run.
  - Without --wipe, uses `git rm --cached` (untrack only, keep file in working tree).
  - With --wipe, uses `git rm` (also removes from working tree).
  - Use --no-dry-run to actually perform the removal.

Examples:
  python tools/cleanup_ignored_tracked.py check
  python tools/cleanup_ignored_tracked.py delete               # dry run, untrack only
  python tools/cleanup_ignored_tracked.py delete --no-dry-run  # actually untrack
  python tools/cleanup_ignored_tracked.py delete --no-dry-run --wipe  # remove from index and working tree

Notes:
  - This script shells out to Git and requires being run inside a Git repo.
  - It respects .gitignore and other standard ignore files via `git check-ignore`.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Iterable, List, Tuple


def run(cmd: List[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


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


def find_tracked_and_ignored(repo: str, limit_to_path: str | None = None) -> List[Tuple[str, str]]:
    tracked = get_tracked_files(repo)
    if limit_to_path:
        limit_to_path = os.path.normpath(limit_to_path)
        tracked = [p for p in tracked if p == limit_to_path or p.startswith(limit_to_path.rstrip(os.sep) + os.sep)]
    return filter_ignored(repo, tracked)


def print_list(items: List[Tuple[str, str]]) -> None:
    if not items:
        print("No tracked files are ignored by Git. ✅")
        return

    print("Tracked files currently ignored by Git (source:line:pattern -> path):")
    for path, meta in items:
        print(f"  {meta} -> {path}")
    print(f"\nTotal: {len(items)} file(s)")


def delete_items(repo: str, items: List[Tuple[str, str]], dry_run: bool, wipe: bool) -> int:
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
        print("\nDry run complete. Re-run with --no-dry-run to apply.")
    else:
        print("\nRemoval complete. Remember to commit the changes.")
    return exit_code


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage tracked files that are ignored by Git")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--path",
        default=None,
        help="Limit operation to a specific path (file or directory) relative to repo root",
    )

    sub.add_parser("check", parents=[common], help="List tracked files that are ignored by Git")

    p_delete = sub.add_parser(
        "delete",
        parents=[common],
        help="Remove tracked files that are ignored by Git (dry-run by default)",
    )
    p_delete.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually perform the removals (default is dry-run)",
    )
    p_delete.add_argument(
        "--wipe",
        action="store_true",
        help="Also remove files from the working tree (uses 'git rm' instead of 'git rm --cached')",
    )

    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    repo = git_root()

    items = find_tracked_and_ignored(repo, limit_to_path=args.path)

    if args.mode == "check":
        print_list(items)
        return 0

    if args.mode == "delete":
        dry_run = not getattr(args, "no_dry_run", False)
        wipe = getattr(args, "wipe", False)
        return delete_items(repo, items, dry_run=dry_run, wipe=wipe)

    print(f"Unknown mode: {args.mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

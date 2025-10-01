from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


FileState = Tuple[bool, float, int]


def snapshot_file_states(paths: Iterable[Path]) -> Dict[Path, FileState]:
    """Capture existence, modification time, and size for each path."""

    snapshot: Dict[Path, FileState] = {}
    for path in paths:
        try:
            if path.exists():
                stat = path.stat()
                snapshot[path] = (True, stat.st_mtime, stat.st_size)
            else:
                snapshot[path] = (False, 0.0, 0)
        except OSError:
            snapshot[path] = (False, 0.0, 0)
    return snapshot


def diff_file_states(
    paths: Iterable[Path], before: Dict[Path, FileState]
) -> Tuple[Set[Path], Set[Path], Set[Path]]:
    """Compare file states and determine created, updated, and skipped paths."""

    after = snapshot_file_states(paths)
    created: Set[Path] = set()
    updated: Set[Path] = set()
    skipped: Set[Path] = set()

    for path in set(before.keys()) | set(after.keys()):
        b_exists, b_mtime, b_size = before.get(path, (False, 0.0, 0))
        a_exists, a_mtime, a_size = after.get(path, (False, 0.0, 0))

        if not b_exists and a_exists:
            created.add(path)
        elif b_exists and not a_exists:
            continue
        elif b_exists and a_exists:
            if b_mtime != a_mtime or b_size != a_size:
                updated.add(path)
            else:
                skipped.add(path)

    return created, updated, skipped


__all__ = [
    "FileState",
    "snapshot_file_states",
    "diff_file_states",
]

#!/usr/bin/env python3
import sys
import subprocess
import signal
from typing import List


def run_with_timeout(timeout_s: int, cmd: List[str]) -> int:
    # Use SIGALRM when available (Unix), else fallback to subprocess timeout
    def handler(signum, frame):
        raise TimeoutError

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_s)
        try:
            proc = subprocess.run(cmd, check=False)
            return proc.returncode
        finally:
            signal.alarm(0)
    except TimeoutError:
        return 124
    except (AttributeError, ValueError, OSError):
        # Fallback: no SIGALRM (e.g., Windows) or not supported
        try:
            proc = subprocess.run(cmd, timeout=timeout_s, check=False)
            return proc.returncode
        except subprocess.TimeoutExpired:
            return 124


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: with_timeout.py <seconds> <cmd> [args...]", file=sys.stderr)
        return 2
    try:
        timeout_s = int(argv[0])
    except ValueError:
        print("First argument must be an integer number of seconds", file=sys.stderr)
        return 2
    cmd = argv[1:]
    if not cmd:
        print("Provide a command to execute", file=sys.stderr)
        return 2
    return run_with_timeout(timeout_s, cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

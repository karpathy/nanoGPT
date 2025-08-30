import os
import sys
import time
import signal
import subprocess
from pathlib import Path

import pytest

pytest.importorskip(
    "lit_nlp",
    reason="lit-nlp dev dependency not installed; skipping 'analyze' smoke test",
)


def test_cli_analyze_bundestag_char_starts_and_stops(tmp_path: Path):
    """Smoke test: start the analyze server and ensure it prints startup info.

    Uses a subprocess to avoid blocking the test process. We don't hit HTTP;
    detecting the startup banner is sufficient.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "ml_playground.cli",
        "analyze",
        "bundestag_char",
        "--port",
        "0",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[1]),  # project root
        env=env,
        text=True,
    )

    assert proc.stdout is not None

    started = False
    start_deadline = time.time() + 60
    output_accum = []
    try:
        while time.time() < start_deadline and proc.poll() is None:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            output_accum.append(line)
            if "[LIT] Starting server" in line:
                started = True
                break
    finally:
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    proc.send_signal(signal.SIGINT)
                else:
                    proc.terminate()
            except Exception:
                pass
        deadline = time.time() + 10
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        if proc.poll() is None:
            proc.kill()
        # Ensure we close the pipe to avoid ResourceWarning on some Python versions
        try:
            if proc.stdout is not None and not proc.stdout.closed:
                proc.stdout.close()
        except Exception:
            pass

    all_out = "".join(output_accum)
    assert started, f"Analyze server failed to start; output was:\n{all_out}"
    assert "[LIT] Registered models:" in all_out
    assert "[LIT] Registered datasets:" in all_out

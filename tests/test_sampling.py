import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Train a very small model for ~30 iters into a temp out_dir, return that dir."""
    if not (REPO_ROOT / "data" / "shakespeare_char" / "train.bin").exists():
        pytest.skip("shakespeare_char data not prepared")

    out_dir = tmp_path_factory.mktemp("ckpt")
    cmd = [
        sys.executable,
        "train.py",
        "config/train_shakespeare_char.py",
        "--compile=False",
        "--max_iters=30",
        "--eval_interval=15",
        "--eval_iters=5",
        "--log_interval=30",
        "--n_layer=2",
        "--n_head=2",
        "--n_embd=64",
        "--block_size=64",
        "--batch_size=8",
        f"--out_dir={out_dir}",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"train failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (out_dir / "ckpt.pt").exists(), "checkpoint not written"
    return out_dir


def test_sample_runs_against_tiny_checkpoint(tiny_checkpoint):
    cmd = [
        sys.executable,
        "sample.py",
        f"--out_dir={tiny_checkpoint}",
        "--num_samples=1",
        "--max_new_tokens=20",
        "--device=cuda",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"sample failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Traceback" not in result.stderr
    assert "---------------" in result.stdout, "expected sample separator missing"

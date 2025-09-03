from __future__ import annotations

import pytest
import torch

from ml_playground.cli import _global_device_setup


def test_global_setup_sets_seed_and_is_deterministic() -> None:
    # First call with seed=123
    _global_device_setup("cpu", "float32", seed=123)
    a = torch.rand(1).item()
    # Different seed -> different number likely
    _global_device_setup("cpu", "float32", seed=124)
    b = torch.rand(1).item()
    # Same seed again -> reproducible
    _global_device_setup("cpu", "float32", seed=123)
    c = torch.rand(1).item()
    assert a != b
    assert abs(a - c) < 1e-8


def test_global_setup_enables_tf32_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate CUDA availability; ensure flags become True after setup
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=True)
    # Reset flags before
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    _global_device_setup("cuda", "bfloat16", seed=1)
    # Flags should be enabled
    assert getattr(torch.backends.cuda.matmul, "allow_tf32", True) is True
    assert getattr(torch.backends.cudnn, "allow_tf32", True) is True


def test_global_setup_no_crash_without_cuda() -> None:
    # Should not raise even if CUDA unavailable
    _global_device_setup("cuda", "float16", seed=42)

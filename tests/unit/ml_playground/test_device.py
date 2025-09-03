from __future__ import annotations

import types
from contextlib import nullcontext

import pytest
import torch

from ml_playground.device import AmpContext, setup


def test_ampcontext_cpu_fp32_no_autocast(monkeypatch: pytest.MonkeyPatch):
    # Force CPU path
    with AmpContext("cpu", torch.float32):
        # When autocast is disabled, __enter__ may return None; just ensure no error
        pass


def test_setup_selects_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=True)
    device_type, ptdtype, ctx = setup("cuda", "float16", seed=123)
    assert device_type in ("cpu", "mps")  # no cuda -> cpu or mps depending on env
    assert ptdtype in (torch.float16, torch.bfloat16, torch.float32)
    # context is usable
    with ctx:
        pass


def test_setup_prefers_mps_when_requested_and_available(
    monkeypatch: pytest.MonkeyPatch,
):
    # Create a fake mps backend with is_available() -> True
    fake_mps = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch.backends, "mps", fake_mps, raising=False)
    device_type, ptdtype, ctx = setup("mps", "float16", seed=1)
    assert device_type == "mps"
    with ctx:
        pass


def test_setup_cuda_branch(monkeypatch: pytest.MonkeyPatch):
    # Simulate CUDA available without requiring actual GPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=True)
    # Avoid touching real CUDA internals in torch.amp.autocast
    monkeypatch.setattr(
        torch.amp, "autocast", lambda *a, **k: nullcontext(), raising=True
    )
    device_type, ptdtype, ctx = setup("cuda", "bfloat16", seed=1)
    assert device_type == "cuda"
    with ctx:
        pass


def test_ampcontext_mps_non_fp16_uses_nullcontext(monkeypatch: pytest.MonkeyPatch):
    # Simulate MPS available and verify bfloat16 path uses nullcontext branch
    fake_mps = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch.backends, "mps", fake_mps, raising=False)
    with AmpContext("mps", torch.bfloat16):
        # Should enter and exit without trying to enable autocast for non-fp16
        pass

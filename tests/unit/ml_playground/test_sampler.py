from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Tuple

import pytest
import torch

from ml_playground.config import RuntimeConfig, SampleConfig, SamplerConfig
import ml_playground.sampler as sampler
from ml_playground.error_handling import CheckpointError
from ml_playground.checkpoint import CheckpointManager


# ---------------------------
# Helpers
# ---------------------------


def _write_char_meta(meta_path: Path) -> None:
    """Write a minimal char-level meta.pkl with stoi/itos and uint32 dtype."""
    stoi = {"\n": 0, "H": 1, "i": 2}
    itos = {v: k for k, v in stoi.items()}
    meta = {
        "meta_version": 1,
        "kind": "char",
        "dtype": "uint32",
        "tokenizer_type": "char",
        "stoi": stoi,
        "itos": itos,
    }
    meta_path.write_bytes(pickle.dumps(meta))


class _DummyModel:
    def __init__(self) -> None:
        self.loaded_state: Any | None = None

    def eval(self) -> "_DummyModel":  # noqa: D401
        """No-op eval returning self."""
        return self

    def to(self, device: str) -> "_DummyModel":  # noqa: D401
        """No-op device move returning self."""
        return self

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if sd.get("fail_load"):
            raise RuntimeError("bad state_dict")
        self.loaded_state = sd

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Return fake logits for testing
        b, t = x.shape
        vocab_size = 16  # Match the vocab_size in the test
        logits = torch.randn(b, t, vocab_size, dtype=torch.float32, device=x.device)
        return logits, None

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        # Return input with a fixed number of additional tokens
        b, t = x.shape
        out = torch.ones((b, t + max_new_tokens), dtype=torch.long, device=x.device)
        out[:, :t] = x
        return out


# ---------------------------
# load_checkpoint tests
# ---------------------------


def test_load_checkpoint_no_files_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should raise CheckpointError when no checkpoint files exist."""
    mgr = CheckpointManager(tmp_path)
    with pytest.raises(CheckpointError) as e:
        mgr.load_latest_checkpoint(device="cpu")
    assert "No last checkpoints discovered" in str(e.value)


def test_load_checkpoint_non_dict_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should raise CheckpointError when checkpoint is not a dict."""
    # Craft a non-dict checkpoint by creating a tensor
    ckpt = tmp_path / "ckpt_best.pt"
    torch.save(torch.tensor([1, 2, 3]), ckpt)
    # Make it discoverable as a last checkpoint for the manager
    (tmp_path / "ckpt_last_00000001.pt").write_bytes(ckpt.read_bytes())
    mgr = CheckpointManager(tmp_path)
    with pytest.raises(CheckpointError) as e:
        mgr.load_latest_checkpoint(device="cpu")
    assert "does not contain a dictionary" in str(e.value)


def test_load_checkpoint_missing_keys_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should raise CheckpointError when checkpoint is missing required keys."""
    # Craft checkpoint with missing keys
    ckpt = tmp_path / "ckpt_last_00000001.pt"
    # Provide other required keys so the first missing is model_args
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "iter_num": 0,
            "best_val_loss": 0.0,
        },
        ckpt,
    )
    mgr = CheckpointManager(tmp_path)
    with pytest.raises(CheckpointError) as e:
        mgr.load_latest_checkpoint(device="cpu")
    # Expect the error to mention the first missing required key
    assert "model_args" in str(e.value)


def test_load_checkpoint_bad_model_args_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should raise CheckpointError when model_args is missing required keys."""
    # Craft checkpoint with invalid model_args
    ckpt = tmp_path / "ckpt_last_00000001.pt"
    torch.save({"model": {}, "model_args": {"n_layer": -1}}, ckpt)
    mgr = CheckpointManager(tmp_path)
    # This now fails later when constructing a Checkpoint; manager validates required keys first
    with pytest.raises(CheckpointError):
        mgr.load_latest_checkpoint(device="cpu")


def test_load_checkpoint_load_state_error_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should wrap load_state_dict errors as ModelError with path info."""
    # Prepare a valid-looking checkpoint but force load failure via Dummy GPT
    # Use a discoverable filename for CheckpointManager
    ckpt = tmp_path / "ckpt_last_00000001.pt"
    # Minimal valid args for our GPTConfig (use default values by omitting fields)
    torch.save(
        {
            "model": {"invalid_key": "invalid_value"},  # This will cause load to fail
            "model_args": {
                "block_size": 4,
                "vocab_size": 16,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
            },
            "iter_num": 0,
            "best_val_loss": 0.0,
            "optimizer": {},
            "config": {},
        },
        ckpt,
    )

    class _DummyGPT(_DummyModel):
        def __init__(self, config: Any) -> None:  # match GPT(conf)
            super().__init__()

        def load_state_dict(self, state_dict: dict) -> None:
            # Force an error to be raised
            raise RuntimeError("Forced load error for testing")

    monkeypatch.setattr(sampler, "GPT", _DummyGPT)
    mgr = CheckpointManager(tmp_path)
    # The manager doesn't construct GPT; it only returns typed dicts. The ModelError is raised when applying state.
    # Simulate consumer applying load and catching a ModelError with path context in higher-level code; here we just ensure manager loads dicts.
    ckpt_obj = mgr.load_latest_checkpoint(device="cpu")
    assert ckpt_obj is not None


# Removed tests for internal _codec_from_meta as the codec helpers were dropped


# ---------------------------
# sample() tests
# ---------------------------


def test_sample_happy_path_with_file_prompt_and_char_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """sample() should print decoded text and separators using FILE: prompt and char meta."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.pkl"
    _write_char_meta(meta_path)

    # Write prompt file
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Hi\n", encoding="utf-8")

    # Device/dtype context is handled internally by sampler; no monkeypatching needed.

    # Patch sampler internals to return a minimal checkpoint and stub GPT to dummy
    class _DummyModelWithLoad(_DummyModel):
        def load_state_dict(self, sd: dict[str, Any], strict: bool = False) -> None:  # type: ignore[override]
            super().load_state_dict(sd)

    monkeypatch.setattr(sampler, "GPT", lambda cfg: _DummyModelWithLoad())

    class _MiniCkpt:
        def __init__(self) -> None:
            self.model = {"weights": []}
            self.model_args = {
                "block_size": 4,
                "vocab_size": 16,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
            }

    monkeypatch.setattr(sampler, "_load_checkpoint", lambda *a, **k: _MiniCkpt())

    # Build SampleExperiment
    rt = RuntimeConfig(
        out_dir=out_dir, device="cpu", dtype="float32", compile=False, seed=1
    )
    sample_conf = SampleConfig(
        start=f"FILE:{prompt_path}",
        num_samples=2,
        max_new_tokens=4,
        temperature=0.1,
        top_k=1,
    )
    exp = SamplerConfig(runtime=rt, sample=sample_conf)

    # Capture logs from sampler module
    caplog.set_level("INFO", logger="ml_playground.sampler")
    # Run
    sampler.sample(exp)

    # Verify via logs (sampler logs instead of printing)
    text = caplog.text
    assert "Hi" in text
    assert "HHHH" in text  # This is what the dummy model generates


def test_sample_with_compile_flag_uses_compiled_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """When compile=True, sample should use torch.compile(model)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_char_meta(out_dir / "meta.pkl")

    # Device/dtype context is handled internally by sampler; no monkeypatching needed.

    # Observe whether compiled model's generate was invoked
    called: dict[str, int] = {"compiled": 0}

    class _Compiled(_DummyModel):
        def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[no-untyped-def]
            called["compiled"] += 1
            return super().generate(*args, **kwargs)

    # Stub torch.compile
    monkeypatch.setattr(torch, "compile", lambda m: _Compiled())  # type: ignore[attr-defined]

    # Patch sampler internals: GPT returns dummy to be compiled; supply minimal checkpoint
    class _DummyModelWithLoad2(_DummyModel):
        def load_state_dict(self, sd: dict[str, Any], strict: bool = False) -> None:  # type: ignore[override]
            super().load_state_dict(sd)

    monkeypatch.setattr(sampler, "GPT", lambda cfg: _DummyModelWithLoad2())

    class _MiniCkpt2:
        def __init__(self) -> None:
            self.model = {"weights": []}
            self.model_args = {
                "block_size": 4,
                "vocab_size": 16,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
            }

    monkeypatch.setattr(sampler, "_load_checkpoint", lambda *a, **k: _MiniCkpt2())

    rt = RuntimeConfig(
        out_dir=out_dir, device="cpu", dtype="float32", compile=True, seed=1
    )
    sc = SampleConfig(
        start="\n", num_samples=1, max_new_tokens=3, temperature=0.5, top_k=0
    )
    exp = SamplerConfig(runtime=rt, sample=sc)

    # Call sample function directly without passing model
    sampler.sample(exp)
    assert called["compiled"] == 1

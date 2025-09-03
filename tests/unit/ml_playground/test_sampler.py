from __future__ import annotations

import builtins
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Tuple

import pytest
import torch

from ml_playground.config import RuntimeConfig, SampleConfig, SamplerConfig
import ml_playground.sampler as sampler
from ml_playground.error_handling import CheckpointError, DataError, ModelError


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
    with pytest.raises(CheckpointError) as e:
        sampler.load_checkpoint(tmp_path, device="cpu")
    assert "No checkpoint found" in str(e.value)


def test_load_checkpoint_non_dict_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should raise CheckpointError when checkpoint is not a dict."""
    # Craft a non-dict checkpoint by creating a tensor
    ckpt = tmp_path / "ckpt_best.pt"
    torch.save(torch.tensor([1, 2, 3]), ckpt)
    with pytest.raises(CheckpointError) as e:
        sampler.load_checkpoint(tmp_path, device="cpu")
    assert "not a dict" in str(e.value)


def test_load_checkpoint_missing_keys_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should raise CheckpointError when checkpoint is missing required keys."""
    # Craft checkpoint with missing keys
    ckpt = tmp_path / "ckpt_best.pt"
    torch.save({"model": {}}, ckpt)  # missing model_args
    with pytest.raises(CheckpointError) as e:
        sampler.load_checkpoint(tmp_path, device="cpu")
    assert "model" in str(e.value)
    assert "model_args" in str(e.value)


def test_load_checkpoint_bad_model_args_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should raise CheckpointError when model_args is missing required keys."""
    # Craft checkpoint with invalid model_args
    ckpt = tmp_path / "ckpt_best.pt"
    torch.save({"model": {}, "model_args": {"n_layer": -1}}, ckpt)
    with pytest.raises(CheckpointError):
        sampler.load_checkpoint(tmp_path, device="cpu")


def test_load_checkpoint_load_state_error_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should wrap load_state_dict errors as ModelError with path info."""
    # Prepare a valid-looking checkpoint but force load failure via Dummy GPT
    ckpt = tmp_path / "ckpt_best.pt"
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
    with pytest.raises(ModelError) as e:
        sampler.load_checkpoint(tmp_path, device="cpu")
    assert str(ckpt) in str(e.value)  # Ensure path is mentioned


# ---------------------------
# _codec_from_meta tests
# ---------------------------


def test_codec_from_meta_char_success(tmp_path: Path) -> None:
    """Char-level meta.pkl should yield working encode/decode callables."""
    meta_path = tmp_path / "meta.pkl"
    _write_char_meta(meta_path)
    encode, decode = sampler._codec_from_meta(meta_path)
    ids = encode("Hi\n")
    assert isinstance(ids, list)
    assert decode(ids).endswith("\n")


def test_codec_from_meta_missing_meta_version_raises(tmp_path: Path) -> None:
    """meta.pkl without meta_version should raise DataError."""
    meta_path = tmp_path / "meta.pkl"
    meta = {"kind": "char", "dtype": "uint32", "stoi": {"a": 1}, "itos": {1: "a"}}
    meta_path.write_bytes(pickle.dumps(meta))
    with pytest.raises(DataError, match="meta_version"):
        sampler._codec_from_meta(meta_path)


def test_codec_from_meta_unsupported_dtype_raises(tmp_path: Path) -> None:
    """meta.pkl with unsupported dtype should raise DataError."""
    meta_path = tmp_path / "meta.pkl"
    meta = {
        "meta_version": 1,
        "kind": "char",
        "dtype": "float64",
        "stoi": {"a": 1},
        "itos": {1: "a"},
    }
    meta_path.write_bytes(pickle.dumps(meta))
    with pytest.raises(DataError, match="dtype"):
        sampler._codec_from_meta(meta_path)


def test_codec_from_meta_meta_json_tiktoken_without_dep_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """meta.json specifying tiktoken without tiktoken installed should raise DataError."""
    # No meta.pkl, but write meta.json next to it
    meta_path = tmp_path / "meta.pkl"  # missing on purpose
    (tmp_path / "meta.json").write_text(
        '{"kind": "tiktoken", "encoding": "gpt2"}', encoding="utf-8"
    )

    # Force ImportError for tiktoken regardless of environment
    real_import = builtins.__import__

    def _no_tiktoken(name: str, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if name == "tiktoken":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tiktoken)
    with pytest.raises(
        DataError,
        match="required to derive codec|tiktoken is required|No usable dataset meta",
    ):
        sampler._codec_from_meta(meta_path)


def test_codec_from_meta_no_meta_and_no_tiktoken_fallback_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no meta files are present and tiktoken import fails, raise DataError."""
    meta_path = tmp_path / "meta.pkl"  # missing
    # Force ImportError
    real_import = builtins.__import__

    def _no_tiktoken(name: str, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if name == "tiktoken":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tiktoken)
    with pytest.raises(DataError, match="required to derive codec"):
        sampler._codec_from_meta(meta_path)


# ---------------------------
# sample() tests
# ---------------------------


def test_sample_happy_path_with_file_prompt_and_char_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """sample() should print decoded text and separators using FILE: prompt and char meta."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.pkl"
    _write_char_meta(meta_path)

    # Write prompt file
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Hi\n", encoding="utf-8")

    # Fake setup() to avoid device/tensor dtype concerns
    monkeypatch.setattr(
        sampler,
        "setup",
        lambda device, dtype, seed: ("cpu", torch.float32, nullcontext()),
    )

    # Patch load_checkpoint to return a dummy model and fake checkpoint
    dummy = _DummyModel()

    def _fake_load(out: Path, device: str) -> Tuple[_DummyModel, Any]:
        # Create a mock checkpoint object
        class MockCheckpoint:
            pass

        mock_ckpt = MockCheckpoint()
        return dummy, mock_ckpt

    monkeypatch.setattr(sampler, "load_checkpoint", _fake_load)

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

    # Run
    sampler.sample(exp)

    # Verify
    captured = capsys.readouterr()
    # Check that the output contains the expected elements
    assert "Hi" in captured.out
    assert "HHHH" in captured.out  # This is what the dummy model generates


def test_sample_with_compile_flag_uses_compiled_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """When compile=True, sample should use torch.compile(model)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_char_meta(out_dir / "meta.pkl")

    # Fake setup context
    monkeypatch.setattr(
        sampler,
        "setup",
        lambda device, dtype, seed: ("cpu", torch.float32, nullcontext()),
    )

    # Observe whether compiled model's generate was invoked
    called: dict[str, int] = {"compiled": 0}

    class _Compiled(_DummyModel):
        def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[no-untyped-def]
            called["compiled"] += 1
            return super().generate(*args, **kwargs)

    # Stub torch.compile
    monkeypatch.setattr(torch, "compile", lambda m: _Compiled())  # type: ignore[attr-defined]
    # Patch load_checkpoint to return a baseline model that will be "compiled"
    dummy = _DummyModel()

    def _fake_load(out: Path, device: str) -> Tuple[_DummyModel, Any]:
        # Create a mock checkpoint object
        class MockCheckpoint:
            pass

        mock_ckpt = MockCheckpoint()
        return dummy, mock_ckpt

    monkeypatch.setattr(sampler, "load_checkpoint", _fake_load)

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

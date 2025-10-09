from __future__ import annotations

import pickle
from pathlib import Path
import itertools

import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from ml_playground.configuration.models import (
    RuntimeConfig,
    SampleConfig,
    SamplerConfig,
    SharedConfig,
)
from ml_playground.sampling.runner import Sampler
from ml_playground.core.tokenizer import CharTokenizer

_RUN_COUNTER = itertools.count()


class _StubModel:
    def __init__(self) -> None:
        self.generate_calls = 0

    def eval(self) -> "_StubModel":
        return self

    def to(self, device: str) -> "_StubModel":  # noqa: D401
        return self

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = False
    ) -> None:  # noqa: D401
        pass

    def generate(
        self, x: torch.Tensor, max_new_tokens: int, *, temperature: float, top_k: int
    ) -> torch.Tensor:
        self.generate_calls += 1
        b, t = x.shape
        out = (
            torch.arange(t + max_new_tokens, dtype=torch.long).unsqueeze(0).repeat(b, 1)
        )
        return out


def _create_shared(tmp_path: Path) -> tuple[SharedConfig, Path]:
    out_dir = tmp_path / f"out_{next(_RUN_COUNTER)}"
    out_dir.mkdir(exist_ok=True)
    meta = {
        "meta_version": 1,
        "kind": "char",
        "dtype": "uint16",
        "tokenizer_type": "char",
        "stoi": {"A": 1},
        "itos": {1: "A"},
    }
    with (out_dir / "meta.pkl").open("wb") as fh:
        pickle.dump(meta, fh)
    return (
        SharedConfig(
            experiment="hypo",
            config_path=out_dir / "cfg.toml",
            project_home=out_dir,
            dataset_dir=out_dir,
            train_out_dir=out_dir,
            sample_out_dir=out_dir,
        ),
        out_dir,
    )


def _build_sampler(tmp_path: Path, start: str) -> tuple[Sampler, _StubModel]:
    shared, out_dir = _create_shared(tmp_path)

    rt = RuntimeConfig(
        out_dir=out_dir, device="cpu", dtype="float32", compile=False, seed=42
    )
    sample_cfg = SampleConfig(
        start=start, num_samples=1, max_new_tokens=2, temperature=1.0, top_k=0
    )

    def _load_ckpt(**_: object) -> object:
        class _Ckpt:
            model = {"weights": []}
            model_args = {
                "block_size": 4,
                "vocab_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 4,
            }

        return _Ckpt()

    model = _StubModel()
    sampler = Sampler(
        SamplerConfig(
            runtime=rt,
            sample=sample_cfg,
            checkpoint_load_fn=_load_ckpt,
            model_factory=lambda cfg, logger: model,
        ),
        shared,
    )
    sampler.tokenizer = CharTokenizer({"A": 1})
    sampler.model = model
    return sampler, model


@given(
    start=st.text(
        alphabet=st.characters(min_codepoint=65, max_codepoint=67),
        min_size=1,
        max_size=4,
    )
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_sampler_prompt_tensor_cache(tmp_path: Path, start: str) -> None:
    """Sampler should reuse cached prompt tensor for identical prompts."""
    sampler, stub_model = _build_sampler(tmp_path, start)

    sampler.run()
    tensor_before = sampler._prompt_tensor
    assert tensor_before is not None
    sampler.run()

    assert sampler._prompt_tensor is tensor_before
    assert stub_model.generate_calls == sampler.sample_cfg.num_samples * 2

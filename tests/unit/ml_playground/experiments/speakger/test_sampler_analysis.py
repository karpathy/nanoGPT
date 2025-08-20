from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import pytest

# Import the module under test
from ml_playground.experiments.speakger import gemma_finetuning_mps as gm


class DummyTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token_id = 1
        self.pad_token = None

    @classmethod
    def from_pretrained(cls, path: str | Path, use_fast: bool | None = None, **_: Any) -> "DummyTokenizer":
        return cls()

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        # fixed 2-token prompt
        return {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> str:  # noqa: E501
        # return controlled generated text including header-like lines and repetitions
        return (
            "Sprecher: Max Mustermann\n"
            "Thema: Test der Wiederholungen\n"
            "Jahr: 2024\n\n"
            "Hallo Welt.\n"
            "Hallo Welt.\n"
            "Dies ist ein Test.\n"
            "Jahr: 2024\n"
            "1999.\n"
        )


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True, pad_token_id=1, eos_token_id=1)
        self.generation_config = SimpleNamespace(use_cache=True)

    def to(self, device: torch.device) -> "DummyModel":
        return self

    def eval(self) -> None:
        return None

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        # Produce output with +5 new tokens beyond the prompt length
        bsz, in_len = input_ids.shape
        total_len = in_len + 5
        out = torch.ones((bsz, total_len), dtype=torch.long)
        out[:, :in_len] = input_ids
        # new tokens are arbitrary ids (1)
        return out


class DummyBaseModel(DummyModel):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "DummyBaseModel":
        return cls()


class DummyPeftModel:
    @classmethod
    def from_pretrained(cls, base_model: Any, adapters_dir: str | Path) -> DummyModel:  # type: ignore[name-defined]
        return DummyModel()


def write_minimal_config(tmp_path: Path, out_dir: Path) -> Path:
    cfg = f"""
[prepare]
dataset = "gemma_finetuning_mps"
raw_dir = "{(tmp_path / 'raw').as_posix()}"
dataset_dir = "{(tmp_path / 'dataset').as_posix()}"

[train.hf_model]
model_name = "dummy"

[train.data]
dataset_dir = "{(tmp_path / 'dataset').as_posix()}"

[train.runtime]
out_dir = "{out_dir.as_posix()}"
device = "cpu"
dtype = "float32"
seed = 1

[sample.runtime]
out_dir = "{out_dir.as_posix()}"
device = "cpu"
dtype = "float32"
seed = 1
compile = false

[sample.sample]
start = "Hello"
max_new_tokens = 5
temperature = 0.0
top_k = 0
top_p = 1.0
num_samples = 1
"""
    path = tmp_path / "config.toml"
    path.write_text(cfg, encoding="utf-8")
    return path


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sampler_writes_json_stats_and_prints_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    # Arrange directories
    out_dir = tmp_path / "out"
    adapters_dir = out_dir / "adapters" / "best"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    # Write state/best.pt to influence filename
    (out_dir / "state").mkdir(parents=True, exist_ok=True)
    torch.save({"best_val_loss": 3.4015, "iter_num": 100}, out_dir / "state" / "best.pt")

    # Monkeypatch heavy deps to dummy ones
    monkeypatch.setattr(gm, "AutoTokenizer", SimpleNamespace(from_pretrained=DummyTokenizer.from_pretrained))
    monkeypatch.setattr(gm, "AutoModelForCausalLM", DummyBaseModel)
    monkeypatch.setattr(gm, "PeftModel", DummyPeftModel)

    # Create config TOML
    config_path = write_minimal_config(tmp_path, out_dir)

    # Act
    gm.sample_from_toml(config_path)

    # Assert: files created
    samples_dir = out_dir / "samples"
    txts = list(samples_dir.glob("*.txt"))
    jsons = list(samples_dir.glob("*.json"))
    assert len(txts) == 1, "sample text file should be created"
    assert len(jsons) == 1, "JSON stats file should be created"
    # Same basename
    assert txts[0].with_suffix(".json").name == jsons[0].name

    # JSON contents minimal checks
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["dataset"] == "speakger"
    assert data["best_val_loss"] == pytest.approx(3.4015)
    assert "analysis" in data
    a = data["analysis"]
    # Sections present
    assert "header" in a and "lines" in a and "ngrams" in a and "anomalies" in a
    # Header values detected
    assert a["header"]["speaker"] == "Max Mustermann"
    assert a["header"]["year"] == "2024"
    # Printed analysis
    out = capsys.readouterr().out
    assert "[gemma_finetuning_mps] Sample analysis:" in out
    assert "== Lines ==" in out

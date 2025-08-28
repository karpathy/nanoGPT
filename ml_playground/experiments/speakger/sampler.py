from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Any

from ml_playground.config import SamplerConfig
import tomllib
from ml_playground.experiments.protocol import (
    Sampler as _SamplerProto,
    SampleReport,
)

# Expose names for monkeypatching in tests (compat with previous integration)
try:  # pragma: no cover - optional heavy deps
    from transformers import AutoTokenizer as AutoTokenizer  # type: ignore
    from transformers import AutoModelForCausalLM as AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover

    class AutoTokenizer:  # type: ignore[no-redef]
        ...

    class AutoModelForCausalLM:  # type: ignore[no-redef]
        ...


try:  # pragma: no cover - optional peft
    from peft import PeftModel as PeftModel  # type: ignore
except Exception:  # pragma: no cover

    class PeftModel:  # type: ignore[no-redef]
        ...


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config.toml"


def _load_best_stats(out_dir: Path) -> tuple[float | None, int | None]:
    try:
        import torch  # local import to avoid hard dep at import time

        best_path = out_dir / "state" / "best.pt"
        if best_path.exists():
            obj: dict[str, Any] = torch.load(best_path, map_location="cpu")  # type: ignore[no-redef]
            raw_best = obj.get("best_val_loss", None)
            best_val: float | None
            if isinstance(raw_best, (int, float, str)):
                try:
                    best_val = float(raw_best)
                except Exception:
                    best_val = None
            else:
                best_val = None
            raw_iter = obj.get("iter_num", 0)
            try:
                iter_num: int = int(raw_iter)  # type: ignore[arg-type]
            except Exception:
                iter_num = 0
            return best_val, iter_num
    except Exception:
        pass
    return None, None


def _analyze_text(text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header: dict[str, str | None] = {"speaker": None, "topic": None, "year": None}
    # Simple header extraction
    for ln in lines[:5]:
        if ln.lower().startswith("sprecher:"):
            header["speaker"] = ln.split(":", 1)[1].strip() or None
        if ln.lower().startswith("thema:"):
            header["topic"] = ln.split(":", 1)[1].strip() or None
        if ln.lower().startswith("jahr:"):
            header["year"] = ln.split(":", 1)[1].strip() or None
    # Repetition analysis (1-grams)
    from collections import Counter

    cnt = Counter(lines)
    ngrams = {"1gram_top": cnt.most_common(5)}
    # Very lightweight anomalies
    anomalies: list[str] = []
    for ln, c in cnt.items():
        if c > 1 and ln not in anomalies:
            anomalies.append(f"repeated: {ln}")
        if ln.isdigit():
            anomalies.append(f"numeric_line: {ln}")
    return {"header": header, "lines": lines, "ngrams": ngrams, "anomalies": anomalies}


def sample_from_toml(config_path: Path) -> None:
    # Read raw TOML dict for robustness to arbitrary keys used in tests
    with Path(config_path).open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
        cur: Any = d
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    # Resolve out_dir
    out_dir_val = _get(raw, "sample.runtime.out_dir") or _get(
        raw, "train.runtime.out_dir"
    )
    if not out_dir_val:
        raise SystemExit("[speakger] out_dir not specified in config")
    out_dir = Path(out_dir_val)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Model config
    model_name = _get(raw, "train.hf_model.model_name", "dummy")

    # Load model + tokenizer per minimal API, allowing tests to monkeypatch
    tok = AutoTokenizer.from_pretrained(out_dir / "tokenizer", use_fast=True)  # type: ignore[attr-defined]
    base = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[attr-defined]
    model: Any
    try:
        model = PeftModel.from_pretrained(base, out_dir / "adapters" / "best")  # type: ignore[attr-defined]
    except Exception:
        model = base  # type: ignore[assignment]

    prompt = _get(raw, "sample.sample.start", "")
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc.get("input_ids")
    attn = enc.get("attention_mask")
    out = model.generate(input_ids=input_ids, attention_mask=attn)
    text = tok.decode(
        out[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Compose filenames depending on whether best.pt exists
    best_val_loss, iter_num = _load_best_stats(out_dir)
    tag = (
        "best"
        if best_val_loss is not None
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    base_name = f"sample_{tag}"

    # Write .txt
    txt_path = samples_dir / f"{base_name}.txt"
    txt_path.write_text(text, encoding="utf-8")

    # Analyze and write .json
    analysis = _analyze_text(text)
    payload = {
        "dataset": "speakger",
        "best_val_loss": best_val_loss,
        "iter_num": iter_num,
        "analysis": analysis,
    }
    json_path = samples_dir / f"{base_name}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[speakger] Sample analysis:")
    print("== Lines ==", len(analysis["lines"]))


class SpeakGerSampler(_SamplerProto):
    def sample(self, cfg: SamplerConfig) -> SampleReport:  # type: ignore[override]
        sample_from_toml(_config_path())
        return SampleReport(
            created_files=(),
            updated_files=(),
            skipped_files=(),
            messages=(
                f"[speakger] sampled using {_config_path()} with in-module sampler",
            ),
        )

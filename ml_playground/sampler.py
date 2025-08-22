from __future__ import annotations
from pathlib import Path
from typing import Callable, Tuple
import pickle
import torch
from ml_playground.model import GPTConfig, GPT
from ml_playground.config import SampleExperiment, RuntimeConfig
from ml_playground.device import setup


def _load_checkpoint(rt: RuntimeConfig, device: str) -> Tuple[GPT, dict]:
    # Use filenames as specified by RuntimeConfig only (no hardcoded candidates)
    candidates = [
        rt.out_dir / rt.ckpt_best_filename,
        rt.out_dir / rt.ckpt_last_filename,
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"No checkpoint found in {rt.out_dir} (tried: {tried}). "
            "Ensure training has produced checkpoints or configure RuntimeConfig filenames."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint at {ckpt_path} is not a dict; got {type(ckpt).__name__}. "
            "Expected a trainer-produced checkpoint."
        )
    required = {"model", "model_args"}
    missing = required - set(ckpt.keys())
    if missing:
        found = ", ".join(sorted(ckpt.keys()))
        raise ValueError(
            "Incompatible checkpoint format: missing required key(s) "
            f"{sorted(missing)} at {ckpt_path}. "
            "Expected a checkpoint produced by ml_playground.trainer with keys 'model' and 'model_args'. "
            f"Found keys: [{found}]."
        )
    try:
        conf: GPTConfig = GPTConfig(**ckpt["model_args"])  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(
            f"Failed to construct GPTConfig from checkpoint model_args at {ckpt_path}: {e}"
        ) from e
    model: GPT = GPT(conf)
    sd = ckpt["model"]
    try:
        # no legacy prefix rewriting; fail fast if keys don't match model
        model.load_state_dict(sd)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model state_dict from checkpoint at {ckpt_path}: {e}"
        ) from e
    return model, ckpt


def _codec_from_meta(
    meta_path: Path | None,
) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Derive encode/decode callables for sampling.

    Preference order:
    1) meta.pkl (strict, legacy format with kind/encoding)
    2) meta.json (best-effort: may provide hints like kind/encoding)
    3) Fallback to tiktoken 'gpt2' encoding (deterministic), with a clear error if tiktoken is unavailable
    """
    # 1) Try strict meta.pkl first
    if meta_path is not None and meta_path.exists():
        with meta_path.open("rb") as f:
            meta = pickle.load(f)

        # Validate common fields
        mv = meta.get("meta_version")
        if mv is None:
            raise ValueError("Missing required 'meta_version' in meta.pkl")
        kind = meta.get("kind")
        dtype = meta.get("dtype")
        if dtype not in {"uint16", "uint32"}:
            raise ValueError(f"Unsupported or missing meta 'dtype': {dtype!r}")

        if kind == "char":
            stoi = meta["stoi"]
            itos = meta["itos"]
            return (
                lambda s: [stoi[c] for c in s],
                lambda ids: "".join(itos[int(i)] for i in ids),
            )
        elif kind == "char_ngram":
            stoi = meta.get("stoi")
            itos = meta.get("itos")
            n = int(meta.get("ngram_size", 1))
            if not isinstance(stoi, dict) or not isinstance(itos, dict) or n < 1:
                raise ValueError(
                    "Invalid meta for char_ngram: expected stoi/itos dicts and ngram_size >= 1"
                )

            def _enc(s: str) -> list[int]:
                if n <= 1:
                    return [stoi[c] for c in s if c in stoi]
                L = len(s)
                if L < n:
                    return []
                out: list[int] = []
                for i in range(0, L - n + 1):
                    tok = s[i : i + n]
                    idx = stoi.get(tok)
                    if idx is not None:
                        out.append(idx)
                return out

            def _dec(ids: list[int]) -> str:
                toks = [itos[int(i)] for i in ids]
                if not toks:
                    return ""
                out = toks[0]
                for t in toks[1:]:
                    # Overlap decode: append the last char
                    out += t[-1]
                return out

            return (_enc, _dec)
        elif kind == "tiktoken":
            enc_name = meta["encoding"]
            try:
                import tiktoken  # type: ignore
            except Exception as e:  # pragma: no cover - optional dep path
                raise RuntimeError(
                    "tiktoken is required to decode with the provided meta (kind='tiktoken'). "
                    "Install it or provide a char-level meta.pkl."
                ) from e

            enc = tiktoken.get_encoding(enc_name)
            return (
                lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
                lambda ids: enc.decode(list(ids)),
            )
        else:
            raise ValueError(f"Unsupported or missing meta 'kind': {kind!r}")

    # 2) Try meta.json next to meta.pkl (best-effort)
    meta_json = None
    out_dir = None
    if meta_path is not None:
        out_dir = meta_path.parent
        cand = meta_path.with_name("meta.json")
        if cand.exists():
            try:
                import json

                with cand.open("r", encoding="utf-8") as f:
                    meta_json = json.load(f)
            except Exception:
                meta_json = None

    if isinstance(meta_json, dict):
        kind = meta_json.get("kind")
        encoding = meta_json.get("encoding")
        # Support the explicit tiktoken case if provided
        if kind == "tiktoken" and isinstance(encoding, str):
            try:
                import tiktoken  # type: ignore
            except Exception as e:  # pragma: no cover - optional dep path
                raise RuntimeError(
                    "tiktoken is required to decode with meta.json (kind='tiktoken'). Install it or provide meta.pkl."
                ) from e
            enc = tiktoken.get_encoding(encoding)
            return (
                lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
                lambda ids: enc.decode(list(ids)),
            )
        # If meta.json doesn't contain usable hints, proceed to fallback

    # 3) Deterministic fallback to GPT-2 BPE via tiktoken
    try:
        import tiktoken  # type: ignore
    except Exception as e:
        where = out_dir if out_dir is not None else meta_path
        raise FileNotFoundError(
            f"No usable dataset meta found at {where}. Expected meta.pkl or meta.json. "
            "As a fallback we require 'tiktoken' to use GPT-2 BPE ('gpt2') encoding, but it is not installed. "
            "Install tiktoken or provide a valid meta.pkl/meta.json."
        ) from e

    enc = tiktoken.get_encoding("gpt2")
    return (
        lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
        lambda ids: enc.decode(list(ids)),
    )


def sample(exp: SampleExperiment) -> None:
    rt = exp.runtime
    device_type, ptdtype, ctx = setup(rt.device, rt.dtype, rt.seed)

    model, _ = _load_checkpoint(rt, device=device_type)
    model.eval().to(device_type)
    run_model = model
    if rt.compile:
        run_model = torch.compile(model)  # type: ignore[attr-defined,assignment]

    # Require dataset meta next to checkpoint outputs
    meta_path = rt.out_dir / "meta.pkl"
    encode, decode = _codec_from_meta(meta_path)

    start = exp.sample.start
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device_type)[None, ...]

    with torch.no_grad():
        with ctx:
            for _ in range(exp.sample.num_samples):
                y = run_model.generate(  # type: ignore[attr-defined]
                    x,
                    exp.sample.max_new_tokens,
                    temperature=exp.sample.temperature,
                    top_k=exp.sample.top_k,
                )
                print(decode(y[0].tolist()))
                print("---------------")

from __future__ import annotations
from pathlib import Path
from typing import Callable, Tuple
import pickle
import torch
from ml_playground.model import GPTConfig, GPT
from ml_playground.config import SampleExperiment
from ml_playground.device import setup


def _load_checkpoint(out_dir: Path, device: str) -> Tuple[GPT, dict]:
    candidates = [
        out_dir / "ckpt_best.pt",
        out_dir / "ckpt_last.pt",
        out_dir / "ckpt.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No checkpoint found in {out_dir} (tried: {tried})")
    ckpt = torch.load(ckpt_path, map_location=device)
    conf: GPTConfig = GPTConfig(**ckpt["model_args"])
    model: GPT = GPT(conf)
    sd = ckpt["model"]
    # no legacy prefix rewriting; fail fast if keys don't match model
    model.load_state_dict(sd)
    return model, ckpt


def _codec_from_meta(
    meta_path: Path | None,
) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    if meta_path is None or not meta_path.exists():
        raise FileNotFoundError("meta.pkl is required for sampling but was not found")
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
    elif kind == "tiktoken":
        enc_name = meta["encoding"]
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(enc_name)
        return (
            lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
            lambda ids: enc.decode(list(ids)),
        )
    else:
        raise ValueError(f"Unsupported or missing meta 'kind': {kind!r}")


def sample(exp: SampleExperiment) -> None:
    rt = exp.runtime
    device_type, ptdtype, ctx = setup(rt.device, rt.dtype, rt.seed)

    model, _ = _load_checkpoint(rt.out_dir, device=device_type)
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

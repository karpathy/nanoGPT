from __future__ import annotations
from pathlib import Path
from typing import Callable, Tuple
import pickle
import torch
from .model import GPTConfig, GPT
from .config import SampleExperiment
from .device import setup


def _load_checkpoint(out_dir: Path, device: str) -> Tuple[GPT, dict]:
    candidates = [out_dir / "ckpt_best.pt", out_dir / "ckpt_last.pt", out_dir / "ckpt.pt"]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No checkpoint found in {out_dir} (tried: {tried})")
    ckpt = torch.load(ckpt_path, map_location=device)
    conf = GPTConfig(**ckpt["model_args"])
    model = GPT(conf)
    sd = ckpt["model"]
    up = "_orig_mod."
    for k in list(sd.keys()):
        if k.startswith(up):
            sd[k[len(up):]] = sd.pop(k)
    model.load_state_dict(sd)
    return model, ckpt


def _codec_from_meta(meta_path: Path | None) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    if meta_path is not None and meta_path.exists():
        with meta_path.open("rb") as f:
            d = pickle.load(f)
        stoi, itos = d["stoi"], d["itos"]
        return (lambda s: [stoi[c] for c in s], lambda l: "".join(itos[i] for i in l))
    # Try GPT-2 BPE via tiktoken; if unavailable, fallback to UTF-8 byte codec
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("gpt2")
        return (
            lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
            lambda l: enc.decode(l),
        )
    except Exception:
        def encode_bytes(s: str) -> list[int]:
            return list(s.encode("utf-8", errors="ignore"))
        def decode_bytes(l: list[int]) -> str:
            return bytes(int(x) & 0xFF for x in l).decode("utf-8", errors="ignore")
        return encode_bytes, decode_bytes


def sample(exp: SampleExperiment) -> None:
    rt = exp.runtime
    device_type, ptdtype, ctx = setup(rt.device, rt.dtype, rt.seed)

    model, _ = _load_checkpoint(rt.out_dir, device=device_type)
    model.eval().to(device_type)
    if rt.compile:
        model = torch.compile(model)

    # We default to GPT-2 BPE if no dataset meta is explicitly provided alongside outputs
    # If you store meta next to checkpoint in future, it will be picked up automatically
    meta_path = rt.out_dir / "meta.pkl"
    encode, decode = _codec_from_meta(meta_path if meta_path.exists() else None)

    start = exp.sample.start
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device_type)[None, ...]

    with torch.no_grad():
        with ctx:
            for _ in range(exp.sample.num_samples):
                y = model.generate(
                    x,
                    exp.sample.max_new_tokens,
                    temperature=exp.sample.temperature,
                    top_k=exp.sample.top_k,
                )
                print(decode(y[0].tolist()))
                print("---------------")

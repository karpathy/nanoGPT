"""
Sweep attention-only benchmarks across multiple sequence lengths.

Example:
  python mea_sweep_attn.py --Ts=8192,16384,32768,65536,131072,262144 --mode=fwd_bwd --impl=block --kernel=triton --chunk=4096 --dtype=bf16 --iters=1 --warmup=1
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from model import mea_attention


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype {s!r}")


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_cuda(fn, device: str, iters: int, warmup: int):
    for _ in range(warmup):
        fn()
    _sync(device)
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync(device)
    ms = start.elapsed_time(end) / iters
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    return ms, peak_mib


def _time_cpu(fn, iters: int, warmup: int):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    dt = (time.time() - t0) / iters
    return dt * 1000.0, float("nan")


def _parse_int_list(s: str) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip().replace("_", "")
        if not part:
            continue
        out.append(int(part))
    return out


@dataclass
class ResultRow:
    T: int
    sdpa_ms: float
    mea_ms: float
    speedup: float
    sdpa_peak_mib: float
    mea_peak_mib: float


def _print_markdown_table(rows: list[ResultRow], *, title: str) -> None:
    print()
    print(title)
    print()
    print("| T | SDPA (ms) | MEA (ms) | Speedup | SDPA peak (MiB) | MEA peak (MiB) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r.T:,} | {r.sdpa_ms:.2f} | {r.mea_ms:.2f} | {r.speedup:.2f}× | {r.sdpa_peak_mib:.0f} | {r.mea_peak_mib:.0f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--nh", type=int, default=12)
    parser.add_argument("--hs", type=int, default=64)
    parser.add_argument("--Ts", type=str, default="8192,16384,32768,65536,131072,262144")
    parser.add_argument("--mode", type=str, default="fwd_bwd", choices=["fwd", "fwd_bwd"])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--order", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--impl", type=str, default="block", choices=["block", "scan"])
    parser.add_argument("--kernel", type=str, default="triton", choices=["torch", "triton"])
    parser.add_argument("--fp32_accum", action="store_true")
    parser.add_argument("--json_out", type=str, default="", help="optional path to write JSON results")
    args = parser.parse_args()

    Ts = _parse_int_list(args.Ts)
    if not Ts:
        raise ValueError("--Ts must contain at least one integer")

    device = args.device
    dtype = _dtype_from_str(args.dtype)
    require_grad = args.mode == "fwd_bwd"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    timer = _time_cuda if device.startswith("cuda") else _time_cpu

    rows: list[ResultRow] = []
    for T in Ts:
        q = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=dtype, requires_grad=require_grad)
        k = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=dtype, requires_grad=require_grad)
        v = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=dtype, requires_grad=require_grad)
        with torch.no_grad():
            q.mul_(1.0 / math.sqrt(args.hs))

        def sdpa_once():
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            if require_grad:
                y.float().sum().backward()
                q.grad = None
                k.grad = None
                v.grad = None
            return y

        def mea_once():
            y = mea_attention(
                q,
                k,
                v,
                order=args.order,
                impl=args.impl,
                kernel=args.kernel,
                chunk_size=args.chunk,
                fp32_accum=args.fp32_accum,
            )
            if require_grad:
                y.float().sum().backward()
                q.grad = None
                k.grad = None
                v.grad = None
            return y

        ms_sdpa, mem_sdpa = timer(sdpa_once, device, args.iters, args.warmup) if device.startswith("cuda") else timer(sdpa_once, args.iters, args.warmup)
        ms_mea, mem_mea = timer(mea_once, device, args.iters, args.warmup) if device.startswith("cuda") else timer(mea_once, args.iters, args.warmup)
        rows.append(
            ResultRow(
                T=T,
                sdpa_ms=ms_sdpa,
                mea_ms=ms_mea,
                speedup=(ms_sdpa / ms_mea),
                sdpa_peak_mib=mem_sdpa,
                mea_peak_mib=mem_mea,
            )
        )

    title = (
        f"device={device} dtype={dtype} mode={args.mode} B={args.B} nh={args.nh} hs={args.hs} "
        f"mea_impl={args.impl} mea_kernel={args.kernel} order={args.order} chunk={args.chunk} fp32_accum={args.fp32_accum}"
    )
    _print_markdown_table(rows, title=title)

    if args.json_out:
        import json

        payload = {
            "config": vars(args),
            "rows": [asdict(r) for r in rows],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print()
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()


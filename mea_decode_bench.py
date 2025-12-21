"""
Decode-time benchmark: KV-cache softmax attention vs stateful MEA recurrence.

This is *decode only* (forward), meant to reflect long-context generation where
softmax attention has O(T) per-token compute with a KV-cache, while MEA can run
with O(1) state updates per token by carrying compact summaries (P/E).

Example:
  python mea_decode_bench.py --Ts=8192,16384,32768,65536,131072,262144,524288,1048576 --decode=128 --dtype=bf16 --order=2 --fp32_state=0 --json_out=decode.json
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from mea_state import mea_state_alloc, mea_state_step_


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype {s!r}")


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


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_decode_cuda(fn, device: str, warmup: int, iters: int):
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


def _time_decode_cpu(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    dt = (time.time() - t0) / iters
    return dt * 1000.0, float("nan")


def _mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024**2)


def _bytes_per_elem(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype {dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--nh", type=int, default=12)
    parser.add_argument("--hs", type=int, default=64)
    parser.add_argument("--Ts", type=str, default="8192,16384,32768,65536,131072,262144,524288,1048576")
    parser.add_argument("--decode", type=int, default=128, help="number of decode tokens to time")
    parser.add_argument("--iters", type=int, default=3, help="repeat the whole decode loop this many times for timing")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--order", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--fp32_state", type=int, default=0, choices=[0, 1], help="store MEA P/E states in fp32 (1) or dtype (0)")
    parser.add_argument("--json_out", type=str, default="", help="optional path to write JSON results")
    args = parser.parse_args()

    Ts = _parse_int_list(args.Ts)
    if not Ts:
        raise ValueError("--Ts must contain at least one integer")

    device = args.device
    dtype = _dtype_from_str(args.dtype)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    B, nh, hs = args.B, args.nh, args.hs

    # Fixed decode tokens (shared between methods)
    q_raw = torch.randn(B, nh, args.decode, hs, device=device, dtype=dtype)
    k_dec = torch.randn(B, nh, args.decode, hs, device=device, dtype=dtype)
    v_dec = torch.randn(B, nh, args.decode, hs, device=device, dtype=dtype)

    # SDPA applies 1/sqrt(d) scaling internally; MEA matches scaled-dot-product
    # attention by scaling Q explicitly (like the MEA module in model.py).
    q_sdpa = q_raw
    q_mea = q_raw.mul(1.0 / math.sqrt(hs))

    acc_dtype = torch.float32 if args.fp32_state else dtype
    q_mea_acc = q_mea.to(acc_dtype)
    k_dec_acc = k_dec.to(acc_dtype)
    v_dec_acc = v_dec.to(acc_dtype)

    timer = _time_decode_cuda if device.startswith("cuda") else _time_decode_cpu

    rows = []
    for T_ctx in Ts:
        # ---- Softmax decode: SDPA with a growing KV-cache (O(T) per token)
        k_cache = torch.randn(B, nh, T_ctx + args.decode, hs, device=device, dtype=dtype)
        v_cache = torch.randn(B, nh, T_ctx + args.decode, hs, device=device, dtype=dtype)
        k_cache[:, :, T_ctx:, :] = k_dec
        v_cache[:, :, T_ctx:, :] = v_dec

        def softmax_decode_once():
            with torch.inference_mode():
                for t in range(args.decode):
                    L = T_ctx + t + 1
                    q_t = q_sdpa[:, :, t : t + 1, :]
                    _ = F.scaled_dot_product_attention(
                        q_t,
                        k_cache[:, :, :L, :],
                        v_cache[:, :, :L, :],
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                    )

        ms_soft, peak_soft = (
            timer(softmax_decode_once, device, args.warmup, args.iters)
            if device.startswith("cuda")
            else timer(softmax_decode_once, args.warmup, args.iters)
        )
        del k_cache, v_cache
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        # ---- MEA decode: carry compact P/E summaries (O(1) per token)
        state0, _acc_dtype = mea_state_alloc(
            B=B,
            nh=nh,
            d=hs,
            dv=hs,
            device=device,
            dtype=dtype,
            order=args.order,
            fp32_state=bool(args.fp32_state),
        )
        # initialize states with a small random value to emulate "already-prefilled" history
        with torch.no_grad():
            state0.P.normal_(mean=0.0, std=0.02)
            if state0.E is not None:
                state0.E.zero_()

        state_work, _ = mea_state_alloc(
            B=B,
            nh=nh,
            d=hs,
            dv=hs,
            device=device,
            dtype=dtype,
            order=args.order,
            fp32_state=bool(args.fp32_state),
        )

        def mea_decode_once():
            with torch.inference_mode():
                state_work.P.copy_(state0.P)
                if state_work.E is not None and state0.E is not None:
                    state_work.E.copy_(state0.E)

                for t in range(args.decode):
                    _ = mea_state_step_(
                        state_work,
                        q=q_mea_acc[:, :, t, :],
                        k=k_dec_acc[:, :, t, :],
                        v=v_dec_acc[:, :, t, :],
                        order=args.order,
                    )

        ms_mea, peak_mea = (
            timer(mea_decode_once, device, args.warmup, args.iters)
            if device.startswith("cuda")
            else timer(mea_decode_once, args.warmup, args.iters)
        )

        ms_per_tok_soft = ms_soft / args.decode
        ms_per_tok_mea = ms_mea / args.decode
        tok_s_soft = 1000.0 / ms_per_tok_soft
        tok_s_mea = 1000.0 / ms_per_tok_mea

        # theoretical cache sizes
        kv_bytes = 2 * B * nh * (T_ctx + args.decode) * hs * _bytes_per_elem(dtype)
        state_bytes = (B * nh * hs * hs * _bytes_per_elem(acc_dtype)) * (2 if args.order == 2 else 1)

        rows.append(
            {
                "T": int(T_ctx),
                "decode": int(args.decode),
                "softmax_ms_per_tok": float(ms_per_tok_soft),
                "mea_ms_per_tok": float(ms_per_tok_mea),
                "softmax_tok_s": float(tok_s_soft),
                "mea_tok_s": float(tok_s_mea),
                "speedup": float(ms_per_tok_soft / ms_per_tok_mea),
                "softmax_peak_mib": float(peak_soft),
                "mea_peak_mib": float(peak_mea),
                "softmax_kv_cache_mib": _mib(kv_bytes),
                "mea_state_mib": _mib(state_bytes),
            }
        )

        print(
            f"T={T_ctx:,} decode={args.decode} | softmax {ms_per_tok_soft:.3f} ms/tok ({tok_s_soft:,.0f} tok/s) | "
            f"mea {ms_per_tok_mea:.3f} ms/tok ({tok_s_mea:,.0f} tok/s) | speedup {ms_per_tok_soft/ms_per_tok_mea:.2f}×"
        )

    if args.json_out:
        import json

        payload = {
            "config": vars(args),
            "rows": rows,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

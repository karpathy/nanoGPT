"""
Numerics report for MEA implementations.

Goals:
  1) Small-T correctness: compare bf16/fp16 outputs against an fp32 naive masked reference
     and report error statistics ("error range").
  2) Large-T stability: for ultra-long T where an fp32 reference is infeasible, run MEA and
     report finiteness + basic output scale stats.

Example:
  # small-T error range (bf16)
  python mea_numerics_report.py --dtype=bf16 --small_Ts=64,128,256 --order=2 --chunk=64 --json_out=mea_numerics.json

  # large-T stability checks (bf16)
  python mea_numerics_report.py --dtype=bf16 --large_Ts=262144,1048576,2097152 --order=2 --chunk=4096 --skip_small --json_out=mea_stability.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

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


def mea_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, order: int) -> torch.Tensor:
    """Naive O(T^2) MEA reference with a strict causal (lower-triangular) mask."""
    if order not in (0, 1, 2):
        raise NotImplementedError
    B, nh, T, hs = q.shape
    A = q @ k.transpose(-2, -1)  # (B, nh, T, T)
    mask = torch.tril(torch.ones((T, T), device=q.device, dtype=torch.bool))
    A = A.masked_fill(~mask, 0)
    y = v
    if order >= 1:
        term1 = A @ v
        y = y + term1
    if order >= 2:
        term2 = A @ term1
        y = y + 0.5 * term2
    return y


def _error_stats(y: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> dict[str, float]:
    y_f = y.float()
    ref_f = y_ref.float()
    err = (y_f - ref_f).abs()
    flat = err.flatten()
    # Per-element relative error is unstable near zeros; report an L2-relative metric instead.
    l2_ref = torch.linalg.vector_norm(ref_f)
    l2_err = torch.linalg.vector_norm(y_f - ref_f)
    l2_rel = (l2_err / (l2_ref + eps)).item()
    rmse = torch.sqrt(torch.mean((y_f - ref_f) ** 2)).item()

    # quantile uses fp32
    p95 = torch.quantile(flat, 0.95).item()
    p99 = torch.quantile(flat, 0.99).item()
    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rmse": float(rmse),
        "p95_abs": float(p95),
        "p99_abs": float(p99),
        "l2_rel": float(l2_rel),
    }


def _tensor_scale_stats(y: torch.Tensor) -> dict[str, float]:
    y_f = y.float()
    return {
        "abs_max": y_f.abs().max().item(),
        "mean": y_f.mean().item(),
        "std": y_f.std(unbiased=False).item(),
    }


def _error_stats_large(y: torch.Tensor, y_ref: torch.Tensor, *, eps: float = 1e-6) -> dict[str, float]:
    """Large-T error stats without quantiles (quantile can be too expensive at 1M+)."""
    y_f = y.float()
    ref_f = y_ref.float()
    diff = y_f - ref_f
    err = diff.abs()
    l2_ref = torch.linalg.vector_norm(ref_f)
    l2_err = torch.linalg.vector_norm(diff)
    l2_rel = (l2_err / (l2_ref + eps)).item()
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rmse": float(rmse),
        "l2_rel": float(l2_rel),
    }


def _time_ms(fn, device: str) -> float:
    _sync(device)
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync(device)
        return start.elapsed_time(end)
    t0 = time.time()
    fn()
    return (time.time() - t0) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--allow_tf32",
        type=str,
        default="true",
        choices=["true", "false"],
        help="whether to allow TF32 for fp32 matmuls (affects fp32 reference runs)",
    )
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--nh", type=int, default=12)
    parser.add_argument("--hs", type=int, default=64)
    parser.add_argument("--order", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--fp32_accum", type=str, default="both", choices=["true", "false", "both"])
    parser.add_argument("--input_std", type=float, default=0.02, help="stddev for random Q/K/V (use ~0.02 for transformer-like scales)")
    parser.add_argument(
        "--impls",
        type=str,
        default="block_torch,scan_torch,block_triton",
        help="comma-separated subset of: block_torch, scan_torch, block_triton",
    )
    parser.add_argument("--small_Ts", type=str, default="64,128,256")
    parser.add_argument("--large_Ts", type=str, default="262144,1048576,2097152")
    parser.add_argument("--skip_small", action="store_true")
    parser.add_argument("--skip_large", action="store_true")
    parser.add_argument(
        "--compare_large_to_fp32",
        action="store_true",
        help="for large_T: also compute an fp32 MEA output (same impl/kernel) and report error vs that reference",
    )
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    device = args.device
    dtype = _dtype_from_str(args.dtype)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    allow_tf32 = args.allow_tf32 == "true"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    small_Ts = _parse_int_list(args.small_Ts)
    large_Ts = _parse_int_list(args.large_Ts)

    fp32_accum_opts = {"true": [True], "false": [False], "both": [False, True]}[args.fp32_accum]

    # Try to detect Triton availability; fall back silently if not installed.
    try:
        import triton  # noqa: F401

        triton_ok = True
    except Exception:
        triton_ok = False

    want_impls = {s.strip() for s in args.impls.split(",") if s.strip()}
    impls = []
    if "block_torch" in want_impls:
        impls.append({"impl": "block", "kernel": "torch"})
    if "scan_torch" in want_impls:
        impls.append({"impl": "scan", "kernel": "torch"})
    if "block_triton" in want_impls and triton_ok:
        impls.append({"impl": "block", "kernel": "triton"})
    if not impls:
        raise ValueError("no implementations selected (check --impls)")

    results: dict[str, object] = {
        "config": {
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "allow_tf32": allow_tf32,
            "B": args.B,
            "nh": args.nh,
            "hs": args.hs,
            "order": args.order,
            "chunk": args.chunk,
            "fp32_accum": args.fp32_accum,
            "small_Ts": small_Ts,
            "large_Ts": large_Ts,
            "impls": impls,
        },
        "small_T": [],
        "large_T": [],
        "large_T_compare": [],
    }

    # ---- Small T: error vs fp32 reference
    if not args.skip_small:
        torch.manual_seed(0)
        for T in small_Ts:
            if T <= 0:
                continue
            q = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=torch.float32) * args.input_std
            k = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=torch.float32) * args.input_std
            v = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=torch.float32) * args.input_std
            q.mul_(1.0 / math.sqrt(args.hs))

            y_ref = mea_reference(q, k, v, order=args.order)

            row = {
                "T": int(T),
                "ref_dtype": "fp32",
                "ref_scale": _tensor_scale_stats(y_ref),
                "cases": [],
            }
            for fp32_accum in fp32_accum_opts:
                for spec in impls:
                    qd = q.to(dtype)
                    kd = k.to(dtype)
                    vd = v.to(dtype)
                    y = mea_attention(
                        qd,
                        kd,
                        vd,
                        order=args.order,
                        chunk_size=args.chunk,
                        fp32_accum=fp32_accum,
                        impl=spec["impl"],
                        kernel=spec["kernel"],
                    )
                    stats = _error_stats(y, y_ref)
                    row["cases"].append(
                        {
                            "impl": spec["impl"],
                            "kernel": spec["kernel"],
                            "fp32_accum": bool(fp32_accum),
                            "out_scale": _tensor_scale_stats(y),
                            "stats": stats,
                        }
                    )
            results["small_T"].append(row)

    # ---- Large T: stability (finite, scale, time, peak memory)
    if not args.skip_large:
        torch.manual_seed(0)
        for T in large_Ts:
            if T <= 0:
                continue
            # Optional fp32 reference for the same random inputs.
            q32 = k32 = v32 = y_ref32 = None
            if args.compare_large_to_fp32:
                q32 = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=torch.float32) * args.input_std
                k32 = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=torch.float32) * args.input_std
                v32 = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=torch.float32) * args.input_std
                q32 = q32 * (1.0 / math.sqrt(args.hs))
            for fp32_accum in fp32_accum_opts:
                for spec in impls:
                    if spec["kernel"] == "torch" and T >= 1_048_576 and args.chunk >= 4096:
                        # Torch kernel materializes chunk×chunk blocks; avoid surprising OOMs at very large T.
                        continue
                    if args.compare_large_to_fp32:
                        q = q32.to(dtype)
                        k = k32.to(dtype)
                        v = v32.to(dtype)
                    else:
                        q = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=dtype) * args.input_std
                        k = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=dtype) * args.input_std
                        v = torch.randn(args.B, args.nh, T, args.hs, device=device, dtype=dtype) * args.input_std
                        q = q * (1.0 / math.sqrt(args.hs))

                    def _run():
                        return mea_attention(
                            q,
                            k,
                            v,
                            order=args.order,
                            chunk_size=args.chunk,
                            fp32_accum=fp32_accum,
                            impl=spec["impl"],
                            kernel=spec["kernel"],
                        )

                    _sync(device)
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    try:
                        if device.startswith("cuda"):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            y = _run()
                            end.record()
                            _sync(device)
                            ms = start.elapsed_time(end)
                        else:
                            t0 = time.time()
                            y = _run()
                            ms = (time.time() - t0) * 1000.0
                    except RuntimeError as e:
                        results["large_T"].append(
                            {
                                "T": int(T),
                                "impl": spec["impl"],
                                "kernel": spec["kernel"],
                                "fp32_accum": bool(fp32_accum),
                                "ok": False,
                                "error": str(e),
                            }
                        )
                        del q, k, v
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        continue

                    finite = bool(torch.isfinite(y).all().item())
                    scale = _tensor_scale_stats(y)
                    peak_mib = float("nan")
                    if device.startswith("cuda"):
                        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

                    results["large_T"].append(
                        {
                            "T": int(T),
                            "impl": spec["impl"],
                            "kernel": spec["kernel"],
                            "fp32_accum": bool(fp32_accum),
                            "ok": True,
                            "finite": finite,
                            "time_ms": float(ms),
                            "peak_mib": float(peak_mib),
                            "scale": scale,
                        }
                    )

                    if args.compare_large_to_fp32:
                        # Compute reference once per (T, impl/kernel) in fp32 for the same inputs.
                        if y_ref32 is None:
                            y_ref32 = mea_attention(
                                q32,
                                k32,
                                v32,
                                order=args.order,
                                chunk_size=args.chunk,
                                fp32_accum=True,  # no-op for fp32 dtype, but explicit for clarity
                                impl=spec["impl"],
                                kernel=spec["kernel"],
                            )
                        stats = _error_stats_large(y, y_ref32)
                        results["large_T_compare"].append(
                            {
                                "T": int(T),
                                "impl": spec["impl"],
                                "kernel": spec["kernel"],
                                "dtype": str(dtype).replace("torch.", ""),
                                "fp32_accum": bool(fp32_accum),
                                "stats": stats,
                            }
                        )

                    del q, k, v, y
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
            if y_ref32 is not None:
                del y_ref32
            if q32 is not None:
                del q32, k32, v32
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    print(json.dumps(results, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

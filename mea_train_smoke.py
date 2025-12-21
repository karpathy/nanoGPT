"""
End-to-end (tiny) GPT training smoke at long context.

This avoids full-token LM loss at huge T by only computing a loss on the final
position logits (the forward pass still runs the full sequence).

Examples:
  # MEA long-context step
  python mea_train_smoke.py --attn_type=mea --T=262144 --n_layer=2 --n_head=4 --n_embd=256 --mea_kernel=triton --mea_chunk=4096

  # Softmax baseline (can get very slow at ultra-long T)
  python mea_train_smoke.py --attn_type=softmax --T=262144 --n_layer=2 --n_head=4 --n_embd=256
"""

from __future__ import annotations

import argparse
import time

import torch

from model import GPT, GPTConfig


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


def _time_step(step_fn, device: str, warmup: int, iters: int):
    for _ in range(warmup):
        step_fn()
    _sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step_fn()
        end.record()
        _sync(device)
        ms = start.elapsed_time(end) / iters
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        return ms, peak_mib
    else:
        t0 = time.time()
        for _ in range(iters):
            step_fn()
        dt = (time.time() - t0) / iters
        return dt * 1000.0, float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--T", type=int, default=262144)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--attn_type", type=str, default="mea", choices=["softmax", "mea"])
    parser.add_argument("--json_out", type=str, default="", help="optional path to write JSON results")

    # MEA knobs (ignored for softmax)
    parser.add_argument("--mea_order", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--mea_impl", type=str, default="block", choices=["block", "scan"])
    parser.add_argument("--mea_kernel", type=str, default="triton", choices=["torch", "triton"])
    parser.add_argument("--mea_chunk", type=int, default=4096)
    parser.add_argument("--mea_fp32_accum", action="store_true")
    parser.add_argument("--mea_qk_l2norm", action="store_true")
    parser.add_argument("--mea_qk_l2norm_eps", type=float, default=1e-6)
    parser.add_argument("--mea_out_groupnorm", action="store_true")
    parser.add_argument("--mea_out_groupnorm_eps", type=float, default=1e-5)
    parser.add_argument("--mea_out_gate", action="store_true")
    args = parser.parse_args()

    device = args.device
    dtype = _dtype_from_str(args.dtype)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GPTConfig(
        block_size=args.T,
        vocab_size=args.vocab,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=args.bias,
        attn_type=args.attn_type,
        mea_order=args.mea_order,
        mea_impl=args.mea_impl,
        mea_kernel=args.mea_kernel,
        mea_chunk_size=args.mea_chunk,
        mea_scale=True,
        mea_fp32_accum=args.mea_fp32_accum,
        mea_qk_l2norm=args.mea_qk_l2norm,
        mea_qk_l2norm_eps=args.mea_qk_l2norm_eps,
        mea_out_groupnorm=args.mea_out_groupnorm,
        mea_out_groupnorm_eps=args.mea_out_groupnorm_eps,
        mea_out_gate=args.mea_out_gate,
    )

    model = GPT(cfg).to(device=device, dtype=dtype)
    model.train()
    optim = model.configure_optimizers(
        weight_decay=0.0, learning_rate=1e-4, betas=(0.9, 0.95), device_type=("cuda" if device.startswith("cuda") else "cpu")
    )

    if args.compile:
        model = torch.compile(model)

    idx = torch.randint(0, cfg.vocab_size, (args.B, args.T), device=device, dtype=torch.long)
    target_last = torch.randint(0, cfg.vocab_size, (args.B, 1), device=device, dtype=torch.long)

    def step():
        optim.zero_grad(set_to_none=True)
        logits, _ = model(idx, targets=None)  # logits: (B, 1, vocab)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_last.view(-1))
        loss.backward()
        optim.step()
        return loss

    try:
        ms, peak_mib = _time_step(step, device=device, warmup=args.warmup, iters=args.iters)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return

    tokens = args.B * args.T
    tok_per_s = tokens / (ms / 1000.0)
    print(
        f"attn_type={args.attn_type} T={args.T} B={args.B} n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} dtype={dtype} compile={args.compile}"
    )
    if args.attn_type == "mea":
        print(f"  mea: order={args.mea_order} impl={args.mea_impl} kernel={args.mea_kernel} chunk={args.mea_chunk} fp32_accum={args.mea_fp32_accum}")
    print(f"  step: {ms:.2f} ms | {tok_per_s:,.0f} tok/s | peak {peak_mib:.0f} MiB")

    if args.json_out:
        import json

        payload = {
            "config": vars(args),
            "metrics": {
                "step_ms": ms,
                "tok_per_s": tok_per_s,
                "peak_mib": peak_mib,
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

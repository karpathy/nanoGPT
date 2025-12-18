"""
Triton kernels for MEA block attention.

Provides a fused triangular matmul primitive:

  out[i] = sum_j mask(i,j) * (q[i] · k[j]) * x[j]

where mask(i,j) is either causal lower-triangular (j<=i) or upper-triangular (j>=i).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - triton is optional (e.g. CPU-only envs)
    triton = None
    tl = None


@dataclass(frozen=True)
class TriMatmulTuning:
    block_m: int = 64
    block_n: int = 64
    block_k: int = 32
    block_v: int = 64
    num_warps: int = 4
    num_stages: int = 2


def _check_inputs(q: torch.Tensor, k: torch.Tensor, x: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("triton is not available (install a CUDA-enabled PyTorch build)")
    if not (q.is_cuda and k.is_cuda and x.is_cuda):
        raise RuntimeError("Triton MEA kernels require CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or x.ndim != 4:
        raise ValueError("expected q,k,x to have shape (B, nh, T, d)")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != x.shape[:3]:
        raise ValueError(f"expected matching (B, nh, T) for q,k,x; got q={tuple(q.shape)} k={tuple(k.shape)} x={tuple(x.shape)}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"expected q and k to share head dim; got {q.shape[-1]} vs {k.shape[-1]}")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported dtype {q.dtype} (expected fp16/bf16/fp32)")
    if k.dtype != q.dtype or x.dtype != q.dtype:
        raise ValueError("expected q,k,x to have the same dtype for Triton kernel")


if triton is not None:
    @triton.jit
    def _trimatmul_fwd_kernel(
    q_ptr,
    k_ptr,
    x_ptr,
    out_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xt: tl.constexpr,
    stride_xv: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_ov: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    NH: tl.constexpr,
    CAUSAL: tl.constexpr,  # True: j<=i (lower). False: j>=i (upper).
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_v = tl.program_id(2)

        b = pid_bh // NH
        h = pid_bh - b * NH

        m0 = pid_m * BLOCK_M
        v0 = pid_v * BLOCK_V

        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_v = v0 + tl.arange(0, BLOCK_V)

        # output accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)

        # load Q tile (BLOCK_M, D) in K-loop
        q_base = q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt

        # Static loop over all key blocks; masking handles tril/triu.
        n_blocks = tl.cdiv(T, BLOCK_N)
        for pid_n in range(0, n_blocks):
            n0 = pid_n * BLOCK_N
            offs_n = n0 + tl.arange(0, BLOCK_N)

            # scores = Q @ K^T (BLOCK_M, BLOCK_N)
            scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, D, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)

                q_ptrs = q_base + offs_k[None, :] * stride_qd
                q = tl.load(q_ptrs, mask=(offs_m[:, None] < T) & (offs_k[None, :] < D), other=0.0)

                k_ptrs = k_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kt + offs_k[None, :] * stride_kd
                k = tl.load(k_ptrs, mask=(offs_n[:, None] < T) & (offs_k[None, :] < D), other=0.0)

                scores += tl.dot(q, tl.trans(k), allow_tf32=False)

            if CAUSAL:
                mask = offs_n[None, :] <= offs_m[:, None]
            else:
                mask = offs_n[None, :] >= offs_m[:, None]
            scores = tl.where(mask, scores, 0.0)

            # out += scores @ X (BLOCK_M, BLOCK_V)
            x_ptrs = x_ptr + b * stride_xb + h * stride_xh + offs_n[:, None] * stride_xt + offs_v[None, :] * stride_xv
            x = tl.load(x_ptrs, mask=(offs_n[:, None] < T) & (offs_v[None, :] < DV), other=0.0)

            # cast scores to X dtype for tensor-core dot
            scores_tc = scores.to(x.dtype)
            acc += tl.dot(scores_tc, x, allow_tf32=False)

        out_ptrs = out_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_v[None, :] * stride_ov
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < T) & (offs_v[None, :] < DV))
else:  # pragma: no cover
    _trimatmul_fwd_kernel = None


def triton_trimatmul(
    q: torch.Tensor,
    k: torch.Tensor,
    x: torch.Tensor,
    *,
    causal: bool,
    tuning: TriMatmulTuning = TriMatmulTuning(),
) -> torch.Tensor:
    """
    Compute out = tril(QK^T)X (causal=True) or triu(QK^T)X (causal=False) using Triton.

    Shapes:
      q,k: (B, nh, T, D)
      x  : (B, nh, T, DV)
      out: (B, nh, T, DV) in the same dtype as inputs, accumulated in fp32
    """
    _check_inputs(q, k, x)
    q = q.contiguous()
    k = k.contiguous()
    x = x.contiguous()

    B, nh, T, D = q.shape
    DV = x.shape[-1]
    if D % tuning.block_k != 0:
        # keep kernel simple; fall back to torch when head dim is unexpected
        raise ValueError(f"head dim D={D} must be divisible by block_k={tuning.block_k}")

    out = torch.empty((B, nh, T, DV), device=q.device, dtype=q.dtype)

    grid = (B * nh, triton.cdiv(T, tuning.block_m), triton.cdiv(DV, tuning.block_v))
    _trimatmul_fwd_kernel[grid](
        q,
        k,
        x,
        out,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qt=q.stride(2),
        stride_qd=q.stride(3),
        stride_kb=k.stride(0),
        stride_kh=k.stride(1),
        stride_kt=k.stride(2),
        stride_kd=k.stride(3),
        stride_xb=x.stride(0),
        stride_xh=x.stride(1),
        stride_xt=x.stride(2),
        stride_xv=x.stride(3),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ot=out.stride(2),
        stride_ov=out.stride(3),
        T=T,
        D=D,
        DV=DV,
        NH=nh,
        CAUSAL=causal,
        BLOCK_M=tuning.block_m,
        BLOCK_N=tuning.block_n,
        BLOCK_K=tuning.block_k,
        BLOCK_V=tuning.block_v,
        num_warps=tuning.num_warps,
        num_stages=tuning.num_stages,
    )
    return out


class TritonTriMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor, causal: bool) -> torch.Tensor:
        if not q.is_cuda:
            raise RuntimeError("TritonTriMatmul requires CUDA tensors")
        out = triton_trimatmul(q, k, x, causal=causal)
        ctx.save_for_backward(q, k, x)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, x = ctx.saved_tensors
        causal = ctx.causal
        dout = dout.contiguous()

        # Let A = mask(QK^T) with mask = tril (causal) or triu (anti-causal).
        # out = A X. With gA = dOut X^T, and gS = mask(gA):
        #   dQ = gS K  = mask(dOut X^T) K
        #   dK = gS^T Q = mask(dOut X^T)^T Q
        #   dX = A^T dOut
        if causal:
            dq = triton_trimatmul(dout, x, k, causal=True)   # tril(dOut X^T) K
            dk = triton_trimatmul(x, dout, q, causal=False)  # triu(X dOut^T) Q
            dx = triton_trimatmul(k, q, dout, causal=False)  # triu(K Q^T) dOut
        else:
            dq = triton_trimatmul(dout, x, k, causal=False)  # triu(dOut X^T) K
            dk = triton_trimatmul(x, dout, q, causal=True)   # tril(X dOut^T) Q
            dx = triton_trimatmul(k, q, dout, causal=True)   # tril(K Q^T) dOut

        return dq, dk, dx, None


def triton_trimatmul_autograd(q: torch.Tensor, k: torch.Tensor, x: torch.Tensor, *, causal: bool) -> torch.Tensor:
    """
    Autograd-enabled wrapper. Returns same dtype as inputs.
    """
    return TritonTriMatmul.apply(q, k, x, causal)

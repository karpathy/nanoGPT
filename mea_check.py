"""
Quick correctness check for the MEA (H<=2) streaming implementation.

Run:
  python mea_check.py
"""

import math
import torch

from model import mea_attention
from mea_state import mea_state_alloc, mea_state_prefill_, mea_state_step_


def mea_reference(q, k, v, order):
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


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    B, nh, T, hs = 2, 3, 32, 8
    q = torch.randn(B, nh, T, hs, device=device, dtype=dtype)
    k = torch.randn(B, nh, T, hs, device=device, dtype=dtype)
    v = torch.randn(B, nh, T, hs, device=device, dtype=dtype)

    q = q * (1.0 / math.sqrt(hs))

    for order in (0, 1, 2):
        y_ref = mea_reference(q, k, v, order=order)
        for impl in ("block", "scan"):
            for chunk_size in (1, 4, 16):
                y = mea_attention(q, k, v, order=order, impl=impl, chunk_size=chunk_size, fp32_accum=True)
                max_err = (y - y_ref).abs().max().item()
                print(f"impl={impl} order={order} chunk={chunk_size} max_err={max_err:.3e}")
                assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)

    # Stateful (decode-style) recurrence check: prefill a prefix, then decode step-by-step.
    B3, nh3, T_ctx, T_dec, hs3 = 2, 3, 9, 7, 8
    q3 = torch.randn(B3, nh3, T_ctx + T_dec, hs3, device=device, dtype=dtype) * (1.0 / math.sqrt(hs3))
    k3 = torch.randn(B3, nh3, T_ctx + T_dec, hs3, device=device, dtype=dtype)
    v3 = torch.randn(B3, nh3, T_ctx + T_dec, hs3, device=device, dtype=dtype)
    for order in (0, 1, 2):
        y_ref = mea_reference(q3, k3, v3, order=order)
        state, acc_dtype = mea_state_alloc(
            B=B3,
            nh=nh3,
            d=hs3,
            dv=hs3,
            device=device,
            dtype=dtype,
            order=order,
            fp32_state=True,
        )
        q3_acc = q3.to(acc_dtype)
        k3_acc = k3.to(acc_dtype)
        v3_acc = v3.to(acc_dtype)
        y_ctx = mea_state_prefill_(state, q=q3_acc[:, :, :T_ctx, :], k=k3_acc[:, :, :T_ctx, :], v=v3_acc[:, :, :T_ctx, :], order=order)
        y_dec = []
        for t in range(T_dec):
            y_t = mea_state_step_(
                state,
                q=q3_acc[:, :, T_ctx + t, :],
                k=k3_acc[:, :, T_ctx + t, :],
                v=v3_acc[:, :, T_ctx + t, :],
                order=order,
            )
            y_dec.append(y_t.unsqueeze(2))
        y_stateful = torch.cat([y_ctx, torch.cat(y_dec, dim=2)], dim=2).to(dtype=dtype)
        max_err = (y_stateful - y_ref).abs().max().item()
        print(f"stateful order={order} max_err={max_err:.3e}")
        assert torch.allclose(y_stateful, y_ref, atol=1e-5, rtol=1e-5)

    # Triton kernel sanity check (forward + backward) vs the torch block reference.
    if device == "cuda":
        try:
            import triton  # noqa: F401
        except Exception:
            triton = None

        if triton is not None:
            # Use a head dim compatible with the Triton kernel tiling.
            B2, nh2, T2, hs2 = 2, 3, 32, 64
            q0 = torch.randn(B2, nh2, T2, hs2, device=device, dtype=dtype) * (1.0 / math.sqrt(hs2))
            k0 = torch.randn(B2, nh2, T2, hs2, device=device, dtype=dtype)
            v0 = torch.randn(B2, nh2, T2, hs2, device=device, dtype=dtype)

            for order in (1, 2):
                for chunk_size in (4, 16):
                    q2 = q0.clone().detach().requires_grad_(True)
                    k2 = k0.clone().detach().requires_grad_(True)
                    v2 = v0.clone().detach().requires_grad_(True)

                    y_torch = mea_attention(q2, k2, v2, order=order, impl="block", kernel="torch", chunk_size=chunk_size, fp32_accum=True)
                    loss = y_torch.float().sum()
                    loss.backward()
                    gq_t, gk_t, gv_t = q2.grad.detach().clone(), k2.grad.detach().clone(), v2.grad.detach().clone()

                    q2.grad = None
                    k2.grad = None
                    v2.grad = None

                    y_triton = mea_attention(q2, k2, v2, order=order, impl="block", kernel="triton", chunk_size=chunk_size, fp32_accum=True)
                    loss = y_triton.float().sum()
                    loss.backward()
                    gq_r, gk_r, gv_r = q2.grad.detach().clone(), k2.grad.detach().clone(), v2.grad.detach().clone()

                    y_err = (y_triton - y_torch).abs().max().item()
                    gq_err = (gq_r - gq_t).abs().max().item()
                    gk_err = (gk_r - gk_t).abs().max().item()
                    gv_err = (gv_r - gv_t).abs().max().item()
                    print(f"triton: order={order} chunk={chunk_size} y_err={y_err:.3e} gq_err={gq_err:.3e} gk_err={gk_err:.3e} gv_err={gv_err:.3e}")
                    assert torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-5)
                    assert torch.allclose(gq_r, gq_t, atol=1e-4, rtol=1e-4)
                    assert torch.allclose(gk_r, gk_t, atol=1e-4, rtol=1e-4)
                    assert torch.allclose(gv_r, gv_t, atol=1e-4, rtol=1e-4)

    print("OK")


if __name__ == "__main__":
    main()

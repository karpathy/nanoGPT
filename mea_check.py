"""
Quick correctness check for the MEA (H<=2) streaming implementation.

Run:
  python mea_check.py
"""

import math
import torch

from model import mea_attention


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
        for chunk_size in (1, 4, 16):
            y = mea_attention(q, k, v, order=order, chunk_size=chunk_size, fp32_accum=True)
            max_err = (y - y_ref).abs().max().item()
            print(f"order={order} chunk={chunk_size} max_err={max_err:.3e}")
            assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)

    print("OK")


if __name__ == "__main__":
    main()


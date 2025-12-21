"""
Stateful (RNN-like) MEA inference utilities.

This implements the order H<=2 asymmetric HLA recurrence:

  y_t = v_t + y_t^(1) + 1/2 y_t^(2)
  y_t^(1) = q_t^T P_t,   P_t = sum_{j<=t} k_j v_j^T
  y_t^(2) = q_t^T E_t,   E_t = sum_{i<=t} k_i (q_i^T P_i)

The key property for decode-time inference is that P (and E for H=2) can be
carried forward as compact states, avoiding an O(T) KV-cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MEAState:
    # (B, nh, d, dv) in accumulation dtype
    P: torch.Tensor
    # (B, nh, d, dv) or None (only needed for order=2)
    E: torch.Tensor | None


def mea_state_alloc(
    *,
    B: int,
    nh: int,
    d: int,
    dv: int,
    device: torch.device | str,
    dtype: torch.dtype,
    order: int,
    fp32_state: bool,
) -> tuple[MEAState, torch.dtype]:
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2 (got {order})")
    acc_dtype = torch.float32 if fp32_state else dtype
    P = torch.zeros((B, nh, d, dv), device=device, dtype=acc_dtype)
    E = torch.zeros_like(P) if order == 2 else None
    return MEAState(P=P, E=E), acc_dtype


def mea_state_step_(
    state: MEAState,
    *,
    q: torch.Tensor,  # (B, nh, d)
    k: torch.Tensor,  # (B, nh, d)
    v: torch.Tensor,  # (B, nh, dv)
    order: int,
) -> torch.Tensor:
    """
    In-place MEA decode update for a single token.

    Returns y of shape (B, nh, dv) in the state's dtype.
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1, or 2 (got {order})")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected q,k,v of shape (B, nh, d)")

    if order == 0:
        return v

    # P <- P + k v^T
    state.P.add_(k.unsqueeze(-1) @ v.unsqueeze(-2))

    y1 = (q.unsqueeze(-2) @ state.P).squeeze(-2)  # (B, nh, dv)
    if order == 1:
        return v + y1

    if state.E is None:
        raise RuntimeError("state.E is required for order=2")
    state.E.add_(k.unsqueeze(-1) @ y1.unsqueeze(-2))
    y2 = (q.unsqueeze(-2) @ state.E).squeeze(-2)
    return v + y1 + 0.5 * y2


@torch.no_grad()
def mea_state_prefill_(
    state: MEAState,
    *,
    q: torch.Tensor,  # (B, nh, T, d)
    k: torch.Tensor,  # (B, nh, T, d)
    v: torch.Tensor,  # (B, nh, T, dv)
    order: int,
) -> torch.Tensor:
    """
    Sequentially update state by scanning a prefix, returning outputs for all positions.

    This is primarily for correctness checks (not optimized).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected q,k,v of shape (B, nh, T, d)")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("expected q,k,v to share (B, nh, T)")
    B, nh, T, _d = q.shape
    dv = v.shape[-1]
    y = torch.empty((B, nh, T, dv), device=q.device, dtype=v.dtype)
    for t in range(T):
        y[:, :, t, :] = mea_state_step_(state, q=q[:, :, t, :], k=k[:, :, t, :], v=v[:, :, t, :], order=order)
    return y

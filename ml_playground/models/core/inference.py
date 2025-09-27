from __future__ import annotations

import torch
from torch.nn import functional as F


def estimate_model_mfu(
    model: torch.nn.Module, fwdbwd_per_iter: int, dt: float
) -> float:
    """Estimate model flops utilization (MFU) relative to A100 peak FLOPs."""

    n_params = sum(p.numel() for p in model.parameters())
    cfg = getattr(model, "config")
    if cfg is None:
        raise AttributeError("model is expected to expose a `config` attribute")

    L = cfg.n_layer
    H = cfg.n_head
    Q = cfg.n_embd // cfg.n_head
    T = cfg.block_size

    flops_per_token = 6 * n_params + 12 * L * H * Q + T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter / dt
    flops_promised = 312e12
    return flops_achieved / flops_promised


def generate_tokens(
    model: torch.nn.Module,
    idx: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively generate tokens from ``model``.

    Args:
        model: The language model with a ``config`` attribute.
        idx: Token indices with shape ``(batch, time)``.
        max_new_tokens: Number of additional tokens to sample.
        temperature: Sampling temperature (0.0 for greedy decoding).
        top_k: Optional nucleus filtering.
    """

    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")

    cfg = getattr(model, "config")
    if cfg is None:
        raise AttributeError("model is expected to expose a `config` attribute")

    max_vocab_idx = cfg.vocab_size - 1

    if torch.any(idx >= cfg.vocab_size):
        idx = torch.clamp(idx, 0, max_vocab_idx)

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx_next = torch.clamp(idx_next, 0, max_vocab_idx)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


__all__ = ["estimate_model_mfu", "generate_tokens"]

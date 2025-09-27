from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

import torch


ParamGroups = Sequence[dict[str, Any]]


class _AdamWFactory(Protocol):
    def __call__(
        self,
        params: Iterable[torch.nn.Parameter] | ParamGroups,
        *,
        lr: float,
        betas: Sequence[float],
        fused: bool | None = None,
    ) -> torch.optim.Optimizer: ...


def configure_optimizers(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
    *,
    factory: _AdamWFactory | None = None,
    logger: Any | None = None,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer matching the legacy GPT implementation."""

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

    optim_groups: ParamGroups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    if logger is not None:
        logger.info(
            "num decayed parameter tensors: %d, with %s parameters",
            len(decay_params),
            f"{sum(p.numel() for p in decay_params):,}",
        )
        logger.info(
            "num non-decayed parameter tensors: %d, with %s parameters",
            len(nodecay_params),
            f"{sum(p.numel() for p in nodecay_params):,}",
        )

    if factory is None:

        def _default_factory(
            params: Iterable[torch.nn.Parameter] | ParamGroups,
            *,
            lr: float,
            betas: Sequence[float],
            fused: bool | None = None,
        ) -> torch.optim.Optimizer:
            kwargs: dict[str, Any] = {"lr": lr, "betas": tuple(betas)}
            if fused is not None:
                kwargs["fused"] = fused
            return torch.optim.AdamW(params, **kwargs)

        factory = _default_factory

    fused_arg: bool | None = True if device_type == "cuda" else None
    return factory(
        optim_groups,
        lr=learning_rate,
        betas=(betas[0], betas[1]),
        fused=fused_arg,
    )


__all__ = ["configure_optimizers"]

from __future__ import annotations


from ml_playground.config import DeviceKind
from ml_playground.model import GPT


class EMA:
    """Maintain an exponential moving average (EMA) of model parameters."""

    def __init__(self, model: GPT, decay: float, device: DeviceKind):
        """Snapshot trainable floating-point parameters onto the target device.

        Args:
            model: The model whose parameters will be tracked.
            decay: EMA decay factor in ``[0, 1)`` controlling smoothing strength.
            device: Target device identifier onto which the shadow weights are moved.
        """
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().to(device)
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: GPT) -> None:
        """Update stored shadow weights with the latest model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype.is_floating_point:
                assert name in self.shadow
                # Standard EMA: new = decay * old + (1 - decay) * current
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.detach()

    def apply_to(self, model: GPT) -> None:
        """Load the EMA weights into ``model`` (typically for evaluation)."""
        model.load_state_dict(self.shadow, strict=False)

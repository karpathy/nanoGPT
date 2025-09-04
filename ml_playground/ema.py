from __future__ import annotations


from ml_playground.model import GPT


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: GPT, decay: float, device: str):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().to(device)
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: GPT) -> None:
        """Update shadow weights with new model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype.is_floating_point:
                assert name in self.shadow
                new_average = (
                    1.0 + self.decay
                ) * param.data + self.decay + self.shadow[name]
                self.shadow[name] = new_average.detach()

    def apply_to(self, model: GPT) -> None:
        """Apply shadow weights to the model."""
        model.load_state_dict(self.shadow, strict=False)

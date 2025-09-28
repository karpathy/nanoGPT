"""Compatibility shim for training APIs.

The core implementation now lives under ``ml_playground.training``. Importing
from this module remains supported for existing integrations pending migration.
"""

from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ml_playground.checkpoint import CheckpointManager
from ml_playground.data import SimpleBatches, _MemmapReader, _sample_batch
from ml_playground.estimator import estimate_loss
from ml_playground.models.core.model import GPT
from ml_playground.training import Trainer, get_lr, train

sample_batch = _sample_batch
MemmapReader = _MemmapReader

__all__ = [
    "Trainer",
    "train",
    "get_lr",
    "SimpleBatches",
    "sample_batch",
    "_sample_batch",
    "MemmapReader",
    "_MemmapReader",
    "GPT",
    "GradScaler",
    "SummaryWriter",
    "CheckpointManager",
    "estimate_loss",
]

import torch
from abc import ABC, abstractmethod

class BaseModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

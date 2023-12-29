import torch
import torch.nn as nn

# Activation variations
class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)

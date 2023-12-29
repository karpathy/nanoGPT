import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, ndim):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / rms * self.gain


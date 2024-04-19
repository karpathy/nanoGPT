import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        bias = config.bias
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        self.gain = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / rms * self.gain

class pRMSNorm(nn.Module):
    """Partial RMS Normalization"""

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        self.gain = nn.Parameter(torch.ones(ndim))
        self.p = config.prmsnorm_pct # percent of elements to use

    def forward(self, x):
        # Calculate the number of elements to use for pRMS
        k = math.ceil(x.size(-1) * self.p)

        # Select the first k elements along the last dimension
        x_part = x[..., :k]

        # Calculate pRMS
        prms = x_part.norm(2, dim=-1, keepdim=True) / math.sqrt(k)

        return x / prms * self.gain

class kRMSNorm(nn.Module):
    """First k elements RMS Normalization"""

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        self.gain = nn.Parameter(torch.ones(ndim))
        self.k = config.krmsnorm_num # percent of elements to use

    def forward(self, x):
        # Calculate the number of elements to use for pRMS
        k = min(x.size(-1), self.k)

        # Select the first k elements along the last dimension
        x_part = x[..., :k]

        # Calculate kRMS
        krms = x_part.norm(2, dim=-1, keepdim=True) / math.sqrt(k)

        return x / krms * self.gain

norm_dictionary = {
    "layernorm": LayerNorm,
    "rmsnorm": RMSNorm,
    "prmsnorm": pRMSNorm,
    "krmsnorm": kRMSNorm,
}

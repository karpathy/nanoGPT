import torch
import torch.nn as nn


# Custom Activation Variations
class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


activation_dictionary = {
    "squared_relu": SquaredReLU(),
    "mish": nn.Mish(),
    "softsign": nn.Softsign(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "rrelu": nn.RReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "glu": nn.GLU(),
    "tanh": nn.Tanh(),
    "selu": nn.SELU(),
    "celu": nn.CELU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "relu6": nn.ReLU6(),
    "prelu": nn.PReLU(),
}

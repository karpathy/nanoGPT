import torch
import torch.nn as nn
import math

# Softmax base 2, with option to remove max subtraction
class Softermax(nn.Module):
    """ Base-2 Softmax with option to remove max subtraction"""
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.subtract_max = config.softermax_use_xmax

    def forward(self, x):
        if self.subtract_max:
            max_x = x.max(dim=self.dim, keepdim=True).values
            x = x - max_x
        e_x = torch.pow(2.0, x)
        return e_x / e_x.sum(dim=self.dim, keepdim=True)

# Softmax variation with learnable constant parameters for xmax and denominator
class Constantmax(nn.Module):
    """ Constant learnable parameters for xmax and denominator """
    def __init__(self, config, dim=-1):
        super().__init__()

        # learnable 'xmax' - beta
        self.beta = nn.Parameter(torch.Tensor([config.constantmax_initial_beta]))

        # denominator - gamma
        self.gamma = nn.Parameter(torch.Tensor([config.constantmax_initial_gamma]))

        # Set the base of the exponent
        if config.constantmax_use_euler_base:
          self.constantmax_base = math.e
        else:
          self.constantmax_base = config.constantmax_base

    def forward(self, x):
        x = x - self.beta
        e_x = torch.pow(self.constantmax_base, x)
        return e_x / self.gamma

# Constantmax Quantized

## Quantization Methods Utilized for Separate Forward and Backward Passes
def quantize(tensor,scale):
    tensor = tensor.mul(scale)
    tensor = torch.round(tensor)
    return tensor
def dequantize(tensor,scale):
    tensor = tensor.div(scale)
    return tensor

## helper class for Constantmax_quan
class const_quan(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""
    @staticmethod
    def forward(ctx, beta=None, gamma=None):
        #scaling factor for beta and gamma while doing quantization
        scale_beta=100 #scaling factor for quantization, should make it as parameter
        scale_gamma=10
        beta = quantize(beta, scale_beta)
        gamma = quantize(gamma, scale_gamma)
        return dequantize(beta, scale_beta),dequantize(gamma,scale_gamma)

    @staticmethod
    def backward(ctx, grad_gamma, grad_beta):
        return grad_gamma, grad_beta

_const_quan=const_quan.apply

class Constantmax_quan(nn.Module):
    """ Base-e Softmax with option to remove max subtraction"""
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim

        # demonimator - gamma
        self.gamma = nn.Parameter(torch.Tensor([config.constantmax_initial_gamma]))

        # learnable 'xmax' - beta
        self.beta = nn.Parameter(torch.Tensor([config.constantmax_initial_beta]))

        self.fake_beta = None
        self.fake_gamma = None

    def forward(self, x):
        if self.training:
            #print('fake_beta:', self.fake_beta)
            #print('fake_gamma:', self.fake_gamma)
            self.fake_beta, self.fake_gamma=_const_quan(self.beta, self.gamma)
            x = x - self.fake_beta
            e_x = torch.exp(x)
            return e_x / self.fake_gamma
        else:
            scale_beta=100 #scaling factor for quantization, should make it as parameter
            scale_gamma=10
            x = x - dequantize(quantize(self.beta,scale_beta), scale_beta)
            e_x = torch.exp(x)
            return e_x/dequantize(quantize(self.gamma,scale_gamma), scale_gamma)

# Like softermax, but parameterized to permit exploration of bases greater than 2
class Strongermax(nn.Module):
    """ Softmax with ability to increase to 'stronger' bases """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.strength = config.strongermax_strength
        self.subtract_max = config.softermax_use_xmax

    def forward(self, x):
        if self.subtract_max:
            max_x = x.max(dim=self.dim, keepdim=True).values
            x = x - max_x
        e_x = torch.pow(self.strength, x)
        return e_x / e_x.sum(dim=self.dim, keepdim=True)

# Using polynomial instead of exponential for Softmax separation non-linearity
class Polymax(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert(config.polymax_x_intercept < 0) # ensure x_intercept is strictly left of the y-axis

        self.x_intercept = config.polymax_x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = config.polymax_y_intercept # where teh graph crosses y-axis

        self.power = config.polymax_power
        self.divisor = config.polymax_divisor

    def forward(self, x):
        # Overview:
        # Flat section:       -inf < x < x_intercept
        # Linear section:     x_intercept <= x <= 0
        # Polynomial section: 0 < x < inf

        # Flat section
        flat_piece = torch.where(x < self.x_intercept, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device))

        # Linear section
        m = self.y_intercept/self.x_intercept # aka 'slope', also x intercept !=0
        b = self.y_intercept
        linear_piece = torch.where((x >= self.x_intercept) & (x <= 0), m * x + b, torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x > 0, x**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        return (poly_piece + linear_piece + flat_piece)/self.divisor

# SigSoftmax from https://arxiv.org/abs/1805.10829
class SigSoftmax(nn.Module):
    """ Softmax variant based on arxiv 1805.10829 with added handles for base """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.base = config.sigsoftmax_base

        # Set the base of the exponent
        if config.sigsoftmax_use_euler_base:
          self.sigsoftmax_base = math.e
        else:
          # custom base
          self.sigsoftmaxmax_base = config.sigsoftmax_base

    def forward(self, inputs):

        # Set exponent
        exp_x = torch.pow(self.sigsoftmax_base, inputs)

        # Similarly set sigmoid approximation
        sig_x = 1 / (1 + torch.pow(self.sigsoftmax_base, -inputs))

        # calculation of numerator and denominator
        numerator = exp_x * sig_x
        denominator = torch.sum(exp_x * sig_x, dim=self.dim, keepdim=True)

        return numerator / denominator


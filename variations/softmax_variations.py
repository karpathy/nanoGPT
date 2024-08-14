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
class ConSmax(nn.Module):
    """ Constant learnable parameters for xmax and denominator """
    def __init__(self, config, dim=-1):
        super().__init__()

        # Input and Output Logging
        self.inputs = []
        self.outputs = []

        # learnable 'xmax' - beta
        self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))

        # denominator - gamma
        self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))

        # Set the base of the exponent
        if config.consmax_use_euler_base:
          self.consmax_base = math.e
        else:
          self.consmax_base = config.consmax_base

    def forward(self, x):
        self.inputs = x
        x = x - self.beta
        e_x = torch.pow(self.consmax_base, x)
        outputs = e_x / self.gamma
        self.outputs = outputs
        return outputs

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

class ConSmaxQuan(nn.Module):
    """ Base-e Softmax with option to remove max subtraction"""
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim

        # demonimator - gamma
        self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))

        # learnable 'xmax' - beta
        self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))

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
        self.subtract_max = config.strongermax_use_xmax
        self.sum_to_1 = config.strongermax_sum_to_1
        self.divisor = config.strongermax_divisor
        self.inputs = []
        self.outputs = []
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):
        self.inputs = x
        if self.subtract_max:
            max_x = x.max(dim=self.dim, keepdim=True).values
            x = x - max_x

        result = torch.pow(self.strength, x)

        if self.sum_to_1:
            result = result / result.sum(dim=self.dim, keepdim=True)

        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        result = result / self.divisor
        self.outputs = result

        return result

# Using polynomial instead of exponential for Softmax separation non-linearity
class Polymax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        assert(config.polymax_x_intercept < 0) # ensure x_intercept is strictly left of the y-axis

        self.dim = dim

        self.div_by_seq_len = config.div_by_seq_len

        self.x_intercept = config.polymax_x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = config.polymax_y_intercept # where the graph crosses y-axis
        self.linear_slope = (self.y_intercept - 0)/(0 - self.x_intercept) # aka 'slope', also x intercept !=0

        self.power = config.polymax_power
        self.divisor = config.polymax_divisor
        self.inputs = []
        self.outputs = []

    def forward(self, x):
        # Overview:
        # Flat section:       -inf < x < x_intercept
        # Linear section:     x_intercept <= x <= 0
        # Polynomial section: 0 < x < inf
        self.inputs = x
        # Flat section
        flat_piece = torch.where(x < self.x_intercept, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device))

        # Linear section
        linear_piece = torch.where((x >= self.x_intercept) & (x <= 0), self.linear_slope * x + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x > 0, x**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + linear_piece + flat_piece)/self.divisor

        # Divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        self.outputs = result

        return result

class VPolymax(nn.Module):
    """ variation of polymax with a v-shape, and is non-monotonically increasing"""
    def __init__(self, config, dim=-1):
        super().__init__()

        assert(config.polymax_x_intercept < 0) # ensure x_intercept is strictly left of the y-axis
        self.dim = dim
        self.div_by_seq_len = config.div_by_seq_len

        self.x_intercept = config.polymax_x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = config.polymax_y_intercept # where the graph crosses y-axis
        self.linear_slope = (self.y_intercept - 0)/(self.x_intercept - 0) # vpoly uses reverse slope

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
        linear_piece = torch.where((x >= self.x_intercept) & (x <= 0), self.linear_slope * x + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x > 0, x**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + linear_piece + flat_piece)/self.divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# Merging of ConSmax body for gradient prop and Polymax head for numerical stability
class SaturatingConSmax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        self.dim = dim

        if config.consmax_learnable_beta:
            # learnable 'xmax' is beta
            self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))
        else:
            self.beta = config.consmax_initial_beta

        if config.consmax_learnable_gamma:
            # denominator is gamma
            self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))
        else:
            self.gamma = config.consmax_initial_gamma

        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

        self.div_by_seq_len = config.div_by_seq_len

        # ConSmax saturation is like ReLU6 but happens where e^x normally would overflow
        # Since we're subtracting x by beta, we only need to guard at "beta + x_sat_value)
        # Note: for e^x this is around 11 for fp16 precision
        self.x_sat = config.consmax_saturation + config.consmax_initial_beta

    def forward(self, x):
        # Overview:
        # exponential section:    -inf < x < (sat_point)
        # flat section:           (sat_point) <= x < inf

        # Exponential section
        exponential_piece = torch.where(
            (x < (self.x_sat)),
            torch.pow(self.consmax_base, x - self.beta),
            torch.tensor(0.0, device=x.device))

        # flat section
        flat_piece = torch.where(x >= (self.x_sat), torch.tensor(self.x_sat, device=x.device), torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (exponential_piece + flat_piece)/self.gamma

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# Merging of ConSmax body for gradient prop and Polymax head for numerical stability
class ExpPolymax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        self.dim = dim

        self.div_by_seq_len = config.div_by_seq_len

        # Base selection
        if config.exppolymax_use_euler_base:
            self.exppolymax_base = math.e
        else:
            self.exppolymax_base = config.exppolymax_base

        self.y_intercept = config.exppolymax_y_intercept # where the graph crosses y-axis
        self.power = config.exppolymax_power
        self.divisor = config.exppolymax_divisor
        # Assumes Euler Base:
        # Shift of x to move poly portion forward to obtain continuous derivative at x=0
        # derivative of poly at 0 should equal a^0
        # d(x^n + y-int) = d(a^x|x=0) = ln(a) * a^0 = ln(a)
        # n * x^(n-1) = ln(a)
        # x = (ln(a) * ( 1 / n )) ** (1/(n-1))
        # Note: if n==1 (straight line) match is already attained, and calculation would nan, so test this case first
        if config.exppolymax_power == 1.0:
            # Note: this only works with y=x and e^x, since we'd have to implement a multiplier or shift teh exponent otherwise.
            self.x_derivative_match_shift = 0
        elif config.exppolymax_use_euler_base:
            # ln(e) = 1
            self.x_derivative_match_shift = (1.0 / config.exppolymax_power)**(1/(config.exppolymax_power - 1))
        else:
            # ln(a) must be calculated, note torch.log is the natural log 'ln'
            self.x_derivative_match_shift = (torch.log(config.exppolymax_base) * (1.0 / config.exppolymax_power))**(1/(config.exppolymax_power - 1))

    def forward(self, x):
        # Overview:
        # exponential section:    -inf < x < 0
        # Polynomial section:     0 < x < inf

        # Exponential section
        exponential_piece = torch.where((x < 0), torch.pow(self.exppolymax_base, x), torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x >= 0, (x + self.x_derivative_match_shift)**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + exponential_piece)/self.divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


# SigSoftmax from https://arxiv.org/abs/1805.10829
class SigSoftmax(nn.Module):
    """ Softmax variant based on arxiv 1805.10829 with added handles for base """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim

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

class ReLUMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.relumax = nn.ReLU()
        self.relumax_divisor = config.relumax_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = self.relumax(x) / self.relumax_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

class Softplus(nn.Module):
    """ Softmax variant based on arxiv 1805.10829 with added handles for base """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softplus = nn.Softplus()
        self.softplus_divisor = config.softplus_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = self.softplus(x) / self.softplus_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


class Squareplus(nn.Module):
    """Squareplus activation function.
       This is a computation friendly version of softplus
       source: https://arxiv.org/abs/2112.11687
    """

    def __init__(self, config, dim=-1, b=4.0*math.log(2)**2):
        super().__init__()
        self.b = b
        self.squareplus_divisor = config.squareplus_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):
        result = 0.5 * (x + torch.sqrt(x**2 + self.b)) / self.squareplus_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# Note: we use the built in library for regular softmax
softmax_dictionary = {
    "consmax": ConSmax,
    "consmax_quan": ConSmaxQuan,
    "saturatingconsmax": SaturatingConSmax,
    "vpolymax": VPolymax,
    "polymax": Polymax,
    "exppolymax": ExpPolymax,
    "softermax": Softermax,
    "strongermax": Strongermax,
    "sigsoftmax": SigSoftmax,
    "relumax": ReLUMax,
    "softplus": Softplus,
    "squareplus": Squareplus,
}

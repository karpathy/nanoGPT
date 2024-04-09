import torch
import torch.nn as nn
import math

linear_dictionary = {
    "linear": nn.Linear(),
    "bitlinear": BitLinear(),
    "bitlinear_optimized": BitLinearOptimized(),
}


class BitLinear(nn.Linear):
    """PyTorch BitLinear Layer
    Source: https://github.com/Beomi/BitNet-Transformers/tree/main
    Source License: Apache Version 2.0
    """

    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Divide weights into groups
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output


class BitLinearOptimized(nn.Linear):
    """Memory Optimized BitLinear Layer
    Source: https://github.com/Beomi/BitNet-Transformers/tree/main
    Source License: Apache Version 2.0
    """

    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinearOptimized, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5

        # Initialize 1-bit quantized weights and store them as int8
        self.register_buffer(
            "quantized_weights", torch.sign(self.weight.data).to(torch.int8)
        )
        # Clear the original weights to save memory
        del self.weight

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.dequantize_weights()

    @weight.setter
    def weight(self, value):
        # Update the quantized_weights when the weight property is set
        self.quantized_weights.data = torch.sign(value).to(torch.int8)

    def dequantize_weights(self):
        # Convert quantized_weights back to bfloat16 and compute alpha for the weights
        bfloat16_weights = self.quantized_weights.to(torch.bfloat16)
        alpha = bfloat16_weights.mean()
        return bfloat16_weights * alpha

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Dequantize the weights before binarization
        weights = self.dequantize_weights()

        # Divide weights into groups
        group_size = weights.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(weights)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = weights[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output

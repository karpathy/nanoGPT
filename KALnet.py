import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

class KAL_Net(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, polynomial_order=3, base_activation=nn.SiLU):
        super(KAL_Net, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = base_activation()
        
        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for Legendre expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # Initialize network parameters
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Polynomial weight for handling Legendre polynomial expansions
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            # Layer normalization to stabilize learning and outputs
            self.layer_norms.append(nn.LayerNorm(out_features))

        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_legendre_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]
        
        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)
        
        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        x = x.to(self.base_weights[0].device)
        batch_size, seq_len, feature_dim = x.size()

        for i, (base_weight, poly_weight, layer_norm) in enumerate(zip(self.base_weights, self.poly_weights, self.layer_norms)):
            base_output = F.linear(self.base_activation(x), base_weight)
            
            # Normalize x to range [-1, 1] for Legendre polynomial computation
            x_normalized = 2 * (x - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0]) - 1
            legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
            legendre_basis = legendre_basis.view(batch_size * seq_len, -1)  # Flatten for linear layer

            poly_output = F.linear(legendre_basis, poly_weight)
            poly_output = poly_output.view(batch_size, seq_len, -1)  # Reshape back to match base_output

            x = self.base_activation(layer_norm(base_output + poly_output))

        return x
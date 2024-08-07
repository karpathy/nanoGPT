import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class TopKRouter(nn.Module):
    """ Conventional Softmax Top_k Gating network (router) NN for MoE layers """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        self.moe_router_scheme = config.moe_router_scheme
        self.route_linear = nn.Linear(config.n_embd, config.n_experts)
    
    def forward(self, x):
        logits = self.route_linear(x)

        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf')) 
        
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output= F.softmax(sparse_logits, dim=-1)

        return router_output, indices


class NoisyTopKRouter(nn.Module):
    """ Noisy Top_k Gating network (router) NN for MoE layers """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        self.moe_router_scheme = config.moe_router_scheme
        self.route_linear = nn.Linear(config.n_embd, config.n_experts)
        self.noise_linear = nn.Linear(config.n_embd, config.n_experts)

    def forward(self, x):
        logits = self.route_linear(x)

        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        
        top_k_noisy_logits = noise_logits + noise
        top_k_logits, indices = logits.topk(self.top_k, dim=1)
        
        zeros = torch.full_like(top_k_noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices

router_dictionary = {
    "softmax": TopKRouter,
    "noisy_top_k": NoisyTopKRouter,
}

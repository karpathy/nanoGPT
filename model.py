"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Query and key rescaling
        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.key_scaling = nn.Parameter(
            torch.full(size=(config.n_head, config.n_embd//config.n_head), fill_value=self.sqk_init_scaling)
        )
        self.query_scaling = nn.Parameter(
            torch.full(size=(config.n_head, config.n_embd//config.n_head), fill_value=self.sqk_init_scaling)
        )

        head_dimension = config.n_embd//config.n_head
        self.scaling_constant = 1.0/math.sqrt(head_dimension)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        sinusoidal_pos = get_sinusoidal_embeddings(T, self.n_embd // self.n_head).to(device=q.device)
        q, k = apply_rotary_position_embeddings(sinusoidal_pos, q, k)

        # Normalize each query and key within each head
        q = q/q.norm(dim=-1, keepdim=True)
        query_scaling = self.query_scaling * (self.sqk_init_value/self.sqk_init_scaling)
        q = q * query_scaling.reshape(1, self.n_head, 1, C // self.n_head)

        k = k/k.norm(dim=-1, keepdim=True)
        key_scaling = self.key_scaling * (self.sqk_init_value/self.sqk_init_scaling)
        k = k * key_scaling.reshape(1, self.n_head, 1, C // self.n_head)

        scaling_factor = math.sqrt(k.size(-1))
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # This implementation expects (B, T, nh, hs) inputs for k,q,v
        # TODO autocast seems to be failing
        # y = flash_attn_func(q.transpose(1, 2).to(dtype=torch.bfloat16), k.transpose(1, 2).to(dtype=torch.bfloat16), v.transpose(1, 2).to(dtype=torch.bfloat16),
        #                    dropout_p=0.0, softmax_scale=scaling_factor, causal=True, window_size=(-1, -1),
        #                    alibi_slopes=None, deterministic=True)
        #

        #y = y.contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # NB this is here for an ablation
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                             dropout_p=self.dropout if self.training else 0,
                                                             is_causal=True, scale=scaling_factor)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.c_fc_u    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc_v = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.scale_u = nn.Parameter(torch.full(size=(4 * config.n_embd,), fill_value=self.suv_init_value, requires_grad=True))
        self.scale_v = nn.Parameter(torch.full(size=(4 * config.n_embd,), fill_value=self.suv_init_value, requires_grad=True))

    def forward(self, x):
        u = self.c_fc_u(x)
        v = self.c_fc_v(x)
        # Apply the scaling
        u_scaling = self.scale_u.reshape(1, 1, -1) * (self.suv_init_value/self.suv_init_scaling) * (self.n_embd ** 0.5)
        v_scaling = self.scale_v.reshape(1, 1, -1) * (self.suv_init_value/self.suv_init_scaling) * (self.n_embd ** 0.5)
        u = u * u_scaling
        v = v * v_scaling
        # Compute SwiGLU
        x = u*self.silu(v)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

        # Scaling parameters applied to the eigen learning rates for the attention step
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale

        # Scaling parameters applied to the eigen learning rates for the mlp step
        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale

        # According to the paper we initialize with alpha_scale and use alpha_init to rescale in the forward pass
        self.alpha_attention = nn.Parameter(
            torch.full(size=(config.n_embd,), fill_value=self.attn_alpha_init_scaling, requires_grad=True))

        self.alpha_mlp = nn.Parameter(
            torch.full(size=(config.n_embd,), fill_value=self.mlp_alpha_init_scaling, requires_grad=True))

    def forward(self, x):
        # The forward pass becomes x<- h+alpha_a(h_A-h) = (1-alpha_a)h + alpha_a h_A, the same for the MLP residual step
        # Normalizations of the activations will be differentiable, we introduce them in the forward computation.
        ## Rescale the parameters

        scaled_alpha_attention = torch.abs(self.alpha_attention * (self.attn_alpha_init_value / self.attn_alpha_init_scaling))
        scaled_alpha_mlp = torch.abs(self.alpha_mlp * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling))

        y_att = self.attn(x)
        y_att = y_att/y_att.norm(dim=-1, keepdim=True)
        x = (1.0 - scaled_alpha_attention[None, None, :]) * x + scaled_alpha_attention[None, None, :] * y_att

        scale = x.norm(dim=-1, keepdim=True)
        x = x / scale

        y_mlp = self.mlp(x)
        scale = y_mlp.norm(dim=-1, keepdim=True)
        y_mlp = y_mlp / scale
        x = (1.0 - scaled_alpha_mlp[None, None, :]) * x + scaled_alpha_mlp[None, None, :] * y_mlp

        scale = x.norm(dim=-1, keepdim=True)
        x = x / scale

        return x



@dataclass
class nGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    base_scale_override: Optional[float] = None
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    @property
    def base_scale(self) -> float:
        if self.base_scale_override is None:
            return 1.0 / (self.n_embd ** 0.5)
        else:
            return self.base_scale_override

class nGPT(nn.Module):

    def __init__(self, config: nGPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.sz_init_value = 1.00
        self.sz_init_scaling = config.base_scale

        self.logit_scale = nn.Parameter(torch.full(size=(config.vocab_size,), fill_value=1.0))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        for ix, block in enumerate(self.transformer.h):
            x = block(x)

        logit_scaling = self.logit_scale.reshape(1, 1, -1) * (self.sz_init_value/self.sz_init_scaling)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits*logit_scaling
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = logits*logit_scaling
            loss = None

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

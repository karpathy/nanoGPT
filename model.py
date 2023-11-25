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

# Softmax base 2, with option to remove max subtraction
class Softermax(nn.Module):
    """ Base-2 Softmax with option to remove max subtraction"""
    def __init__(self, dim=-1, subtract_max=True):
        super().__init__()
        self.dim = dim
        self.subtract_max = subtract_max

    def forward(self, x):
        if self.subtract_max:
            max_x = x.max(dim=self.dim, keepdim=True).values
            x = x - max_x
        e_x = torch.pow(2.0, x)
        return e_x / e_x.sum(dim=self.dim, keepdim=True)

# Softmax with learnable parameters for xmax and denominator
class Constantmax(nn.Module):
    """ Softmax with learnable parameters for xmax and denominator """
    def __init__(self, dim=-1, initial_gamma=500):
        super().__init__()
        self.dim = dim

        # learnable 'xmax' - beta
        self.beta = nn.Parameter(torch.Tensor([0.0]))

        # denominator - gamma
        self.gamma = nn.Parameter(torch.Tensor([initial_gamma]))

    def forward(self, x):
        x = x - self.beta
        e_x = torch.exp(x)
        return e_x / self.gamma

# Like softermax, but parameterized to permit exploration of bases greater than 2
class Strongermax(nn.Module):
    """ Base-2 Softmax with option to remove max subtraction"""
    def __init__(self, dim=-1, subtract_max=True, strength=2):
        super().__init__()
        self.strength = strength
        self.dim = dim
        self.subtract_max = subtract_max

    def forward(self, x):
        if self.subtract_max:
            max_x = x.max(dim=self.dim, keepdim=True).values
            x = x - max_x
        e_x = torch.pow(self.strength, x)
        return e_x / e_x.sum(dim=self.dim, keepdim=True)

# Polynomial estimate of Softmax
class Polymax(nn.Module):
    def __init__(self, x_intercept=-100, y_intercept=1, power=2, divisor=1000.0):
        super().__init__()

        assert(x_intercept < 0) # ensure x_intercepts strictly left of the y-axis
        self.x_intercept = x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = y_intercept # where teh graph crosses y-axis

        self.power = power
        self.divisor = divisor

        # TODO: create single location for printing the model settings
        print("Use polymax")

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
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        exp_x = torch.exp(inputs)
        sig_x = torch.sigmoid(inputs)
        numerator = exp_x * sig_x
        denominator = torch.sum(exp_x * sig_x, dim=-1, keepdim=True)
        return numerator / denominator

# Base2 variant of SigSoftmax from https://arxiv.org/abs/1805.10829
class SigSoftmaxBase2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):

        # Base 2 exponent
        exp2_x = torch.pow(2, inputs)

        # Base 2 sigmoid approximation
        sig2_x = 1 / (1 + torch.pow(2, -inputs))

        numerator = exp2_x * sig2_x
        denominator = torch.sum(exp2_x * sig2_x, dim=-1, keepdim=True)

        return numerator / denominator

class RotaryEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd

        # Register frequencies directly as buffers
        self.register_buffer('freq_left', (10000 ** (torch.arange(0, self.dim//2).float() / self.dim//2)))
        self.register_buffer('freq_right',(10000 ** (torch.arange(0, self.dim//2).float() / self.dim//2)))

    def forward(self, x):
        seq_len = x.shape[-2]
        device = x.device

        t = torch.arange(seq_len, device=device)

        # Get separate frequencies for left and right
        freqs_left = torch.einsum('i,j->ij', t, self.freq_left)
        freqs_right = torch.einsum('i,j->ij', t, self.freq_right)

        # Apply frequencies
        x_left, x_right = x[..., :self.dim//2], x[..., self.dim//2:]
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Combine the left and right parts back
        x = torch.cat([x_left, x_right], dim=-1)

        return x

class ShortRope(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n = config.shortrope_length
        self.dim = config.n_embd

        # Generate freqs of size n rather than full dim
        self.register_buffer('freq_left', (10000 ** (torch.arange(0, self.n//2).float() / self.n//2)))
        self.register_buffer('freq_right', (10000 ** (torch.arange(0, self.n//2).float() / self.n//2)))

    def forward(self, x):
        # Step 1: Get the input tensor shape
        batch_size, seq_len, _ = x.shape

        # Step 2: Split the input tensor into unrotated and rotated sections
        x_unrotated = x[..., :-self.n]  # All but the last n dimensions
        x_rotated = x[..., -self.n:]    # Only the last n dimensions

        # Step 3: Generate rotation frequencies
        t = torch.arange(self.n, device=x.device)
        freqs_left = torch.einsum('i,j->ij', t, self.freq_left)
        freqs_right = torch.einsum('i,j->ij', t, self.freq_right)

        # Calculate how many times to repeat freqs along the sequence length
        num_repeats = seq_len // self.n + int(seq_len % self.n != 0)

        # Repeat the frequency tensors to match the sequence length
        freqs_left = freqs_left.repeat(batch_size, num_repeats, 1)
        freqs_right = freqs_right.repeat(batch_size, num_repeats, 1)

        # Trim the excess elements so the freqs tensors match the sequence length
        freqs_left = freqs_left[:, :seq_len, :]
        freqs_right = freqs_right[:, :seq_len, :]

        # Step 4: Process the x_rotated section
        x_left = x_rotated[..., :self.n//2]
        x_right = x_rotated[..., self.n//2:]

        # Apply the cosine and sine rotations
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Invert the order of the right tensor's last dimension and negate it
        x_right = torch.flip(x_right, dims=[-1]) * -1

        # Combine the left and right rotated sections
        x_rotated = torch.cat([x_left, x_right], dim=-1)

        # Step 5: Combine the rotated and unrotated sections
        x = torch.cat([x_unrotated, x_rotated], dim=-1)

        return x


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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # Rotary Positional Embeddings
        self.rotary_emb = None
        if config.use_rotary_embeddings:
            if config.rope_variant == "rope":
                self.rotary_emb = RotaryEmbedding(config)
            if config.rope_variant == "shortrope":
                self.rotary_emb = ShortRope(config)

        # Softmax Variant Selection
        self.use_softmax_variant = config.use_softmax_variant
        self.softmax_variant = config.softmax_variant
        if self.use_softmax_variant == True:
            # TODO: Add softermax support into flashattention from pytorch 2.0
            # Select softmax variant
            self.softmax_variant = config.softmax_variant

            if self.softmax_variant == "softermax":
              self.use_softermax_xmax = config.use_softermax_xmax
              self.softmax_layer = Softermax(subtract_max=self.use_softermax_xmax)

            if self.softmax_variant == "constantmax":
              self.use_softermax_xmax = config.use_softermax_xmax
              self.constantmax_constant = config.constantmax_constant
              self.softmax_layer = Constantmax()

            if self.softmax_variant == "strongermax":
              self.use_softermax_xmax = config.use_softermax_xmax
              self.softmax_layer = Strongermax(subtract_max=self.use_softermax_xmax,strength=config.strongermax_strength)

            if self.softmax_variant == "polymax":
              self.softmax_layer = Polymax()

            if self.softmax_variant == "sigsoftmax":
              self.softmax_layer = SigSoftmax()

            if self.softmax_variant == "sigsoftmax_base2":
              self.softmax_layer = SigSoftmaxBase2()

            self.flash = False
        else:
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        if self.rotary_emb is not None:
          x = self.rotary_emb(x)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if self.use_softmax_variant:
                att = self.softmax_layer(att)
            else:
                att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # TODO: Change name of self.gelu to something like "self.activation_variant"
        if config.use_relu:
          print("Use ReLU")
          self.gelu = nn.ReLU()
        elif config.use_squared_relu:
          print("Use Squared ReLU")
          self.gelu = SquaredReLU()
        else:
          print("Use GELU")
          self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.use_post_ln = config.use_post_ln

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.use_post_ln:
          x = x + self.attn(self.ln_1(x))
          x = x + self.mlp(self.ln_2(x))
        else:
          x = self.ln_1(x + self.attn(x))
          x = self.ln_2(x + self.mlp(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    # Softmax Alternatives and Options
    use_softmax_variant: bool = False
    softmax_variant: str = "softermax" # Choices: "softermax" "sigsoftmax" "sigsoftmax_base2" "polymax" "strongermax" "constantmax"
    use_softermax_xmax: bool = True # Softermax Option active is softermax selected - True: uses (x - x_max) normalization; False: removes normalization (potential overflow)
    constantmax_constant: int = 1000 # denominator to utilize for Constantmax
    strongermax_strength: int = 2 # Softermax Option active is softermax selected - True: uses (x - x_max) normalization; False: removes normalization (potential overflow)

    # Positional Embeddings Variations
    use_rotary_embeddings: bool = True # If True, uses rotary embeddings, else use conventional absolute position encoding
    rope_variant: str = "shortrope" # options: "shortrope", "rope
    shortrope_length: int = 8 # number of embeddings to use in shortrope
    use_abs_pos_embeddings: bool = True # If True, uses rotary embeddings, else use conventional absolute position encoding

    # Structuring Options, remember to compile the model
    use_post_ln: bool = True
    use_pre_ln: bool = False

    # Layernorm Alternatives and Options
    use_rmsnorm: bool = True # Add option for RMSNorm in place of LayerNorm: https://arxiv.org/abs/1910.07467
    use_relu: bool = False #True: relu squared, False: do not utilize
    use_squared_relu: bool = False #True: utilize relu squared, False: do not utilize
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.use_softmax_variant = config.use_softmax_variant
        self.softmax_variant = config.softmax_variant

        if self.use_softmax_variant == True:
            # Select softmax variant
            self.softmax_variant = config.softmax_variant

            if self.softmax_variant == "softermax":
              self.use_softermax_xmax = config.use_softermax_xmax
              self.softmax_layer = Softermax(subtract_max=self.use_softermax_xmax)

            if self.softmax_variant == "constantmax":
              self.use_softermax_xmax = config.use_softermax_xmax
              self.constantmax_constant = config.constantmax_constant
              self.softmax_layer = Constantmax()

            if self.softmax_variant == "strongermax":
              self.use_softermax_xmax = config.use_softermax_xmax
              self.softmax_layer = Strongermax(subtract_max=self.use_softermax_xmax, strength=config.strongermax_strength)

            if self.softmax_variant == "polymax":
              self.softmax_layer = Polymax()

            if self.softmax_variant == "sigsoftmax":
              self.softmax_layer = SigSoftmax()

            if self.softmax_variant == "sigsoftmax_base2":
              self.softmax_layer = SigSoftmaxBase2()

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = None
        if self.config.use_abs_pos_embeddings:
          pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
          x = self.transformer.drop(tok_emb + pos_emb)
        else:
          x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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

            probs = None
            if self.config.use_softmax_variant:
                probs = self.softmax_layer(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            assert probs != None
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

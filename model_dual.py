"""
model_dual.py

Dual-source weight-slot GPT for abcGPT.

Each weight in the network is represented by TWO parallel parameter tensors:

    W = alpha * W_s + beta * W_w        with        alpha + beta = 1

where W_s is the "shake" slot and W_w is the "wiki" slot, and alpha is a
single scalar knob in [0, 1] at train time (with beta = 1 - alpha). At
inference the same slider can be extrapolated to [-0.5, 1.5] to expose what
the model does outside the trained range. Under this constraint there is
exactly ONE knob, not two — we are training the BALANCE between the slots,
not their absolute magnitudes.

During training:

  - The forward pass materializes W (and b) from the current (alpha, beta) and
    runs a standard F.linear / F.embedding / F.layer_norm with that materialized
    tensor. Autograd therefore routes
        dL/dW_s = alpha * dL/dW
        dL/dW_w = beta  * dL/dW
    for free, without any manual chain-rule code.
  - A `mask_grads(model, corpus)` helper nulls out the gradients on the slot
    that does NOT correspond to the current batch's corpus, BEFORE
    `optimizer.step()`. This is how we keep shake batches from updating wiki
    weights and vice versa.
  - A `set_mix(model, alpha, beta)` helper walks the model and assigns
    `(alpha, beta)` to every Dual* module so the next forward uses that mix.
    The trainer samples alpha from Beta(0.5, 0.5) (or uniform on [0, 1], or
    arcsine) and sets beta = 1 - alpha before each iteration.

Initialization:

  At iter 0 both slots are BYTE-IDENTICAL. We draw W_s with the standard
  nanoGPT init and then copy W_s.data into W_w.data (same for biases). With
  alpha + beta = 1 and W_s == W_w, the materialized weight
      W = alpha * W_s + (1 - alpha) * W_w = W_s
  for ANY alpha in [0, 1]. So at iter 0 the network is exactly the standard
  nanoGPT initialization across the entire slider range — not a degenerate
  zero net at any corner. Training then forces the two slots apart via the
  random alpha sampling combined with the per-corpus gradient routing in
  train_dual.py (shake batches update only W_s, wiki batches update only W_w).

  - Linear: W_s gets nanoGPT's normal_(0, 0.02); bias gets zeros_; then
    W_w.data.copy_(W_s.data) and b_w.data.copy_(b_s.data).
  - Embedding: W_s gets normal_(0, 0.02); then W_w.data.copy_(W_s.data).
  - LayerNorm: W_s = ones, b_s = zeros; then W_w.data.copy_(W_s.data) and
    b_w.data.copy_(b_s.data) (already identical, but kept for uniformity).
  - Residual-projection rescale (std=0.02/sqrt(2*n_layer)) is applied to
    c_proj.W_s and then copied into c_proj.W_w so the byte-identity holds
    after the rescale too.

The class structure (CausalSelfAttention, MLP, Block, GPT) is copied from
model.py and the only edits are module-type substitutions
(nn.Linear -> DualLinear, nn.Embedding -> DualEmbedding, LayerNorm -> DualLayerNorm)
and the weight-tying lines, which tie W_s and W_w of wte to lm_head independently.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Dual modules: each carries two parameter tensors and a current (alpha, beta)
# scalar mix. Forward materializes the effective tensor and calls the standard
# functional op so autograd handles the linear-combination chain rule.
# -----------------------------------------------------------------------------
class DualLinear(nn.Module):
    """Linear layer with dual weight tensors W_s, W_w (and optional dual biases).

    At module-construction time both slots are initialized with PyTorch's
    nn.Linear default scheme (kaiming_uniform_ with a=sqrt(5) for the weight;
    uniform fan-in bound for the bias) and then W_w is set BYTE-IDENTICAL to
    W_s via data.copy_(). DualGPT.__init__ then re-inits everything with the
    standard nanoGPT scheme (normal_(0, 0.02)) and re-copies so that across
    the alpha + beta = 1 slider the iter-0 model equals vanilla nanoGPT init.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # nn.Linear stores weight as (out, in); mirror exactly.
        self.W_s = nn.Parameter(torch.empty(out_features, in_features))
        self.W_w = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.b_s = nn.Parameter(torch.empty(out_features))
            self.b_w = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('b_s', None)
            self.register_parameter('b_w', None)
        # default mix: average (alpha=beta=0.5) so the freshly-initialized
        # module is sensible at evaluation time before any set_mix() call.
        self.alpha = 0.5
        self.beta = 0.5
        # Init W_s with nn.Linear's default scheme, then copy into W_w so the
        # two slots are byte-identical from the start.
        nn.init.kaiming_uniform_(self.W_s, a=math.sqrt(5))
        if bias:
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.b_s, -bound, bound)
        with torch.no_grad():
            self.W_w.data.copy_(self.W_s.data)
            if bias:
                self.b_w.data.copy_(self.b_s.data)

    def forward(self, x):
        W = self.alpha * self.W_s + self.beta * self.W_w
        if self.b_s is not None:
            b = self.alpha * self.b_s + self.beta * self.b_w
        else:
            b = None
        return F.linear(x, W, b)


class DualEmbedding(nn.Module):
    """Embedding layer with dual weight tensors W_s, W_w.

    W_s gets normal(0, 0.02) (nanoGPT default for embeddings); W_w is then
    set byte-identical to W_s. DualGPT may re-init+re-copy after construction."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W_s = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.W_w = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.alpha = 0.5
        self.beta = 0.5
        nn.init.normal_(self.W_s, mean=0.0, std=0.02)
        with torch.no_grad():
            self.W_w.data.copy_(self.W_s.data)

    def forward(self, idx):
        W = self.alpha * self.W_s + self.beta * self.W_w
        return F.embedding(idx, W)


class DualLayerNorm(nn.Module):
    """LayerNorm with dual weight tensors W_s, W_w and dual biases b_s, b_w
    (bias optional, same convention as nanoGPT's LayerNorm)."""

    def __init__(self, ndim, bias=True):
        super().__init__()
        # accept int or tuple, mirror F.layer_norm conventions
        if isinstance(ndim, int):
            normalized_shape = (ndim,)
        else:
            normalized_shape = tuple(ndim)
        self.normalized_shape = normalized_shape
        # gamma=ones, beta=zeros in BOTH slots — byte-identical at init. With
        # alpha+beta=1, the materialized weight is exactly ones (and bias zeros)
        # at any alpha. Gradient routing forces W_s and W_w apart during training.
        self.W_s = nn.Parameter(torch.ones(*normalized_shape))
        self.W_w = nn.Parameter(torch.ones(*normalized_shape))
        if bias:
            self.b_s = nn.Parameter(torch.zeros(*normalized_shape))
            self.b_w = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter('b_s', None)
            self.register_parameter('b_w', None)
        self.alpha = 0.5
        self.beta = 0.5

    def forward(self, x):
        W = self.alpha * self.W_s + self.beta * self.W_w
        if self.b_s is not None:
            b = self.alpha * self.b_s + self.beta * self.b_w
        else:
            b = None
        return F.layer_norm(x, self.normalized_shape, W, b, 1e-5)


# -----------------------------------------------------------------------------
# helpers: walk the model and set (alpha, beta) / null out a slot's gradients.
# Kept deliberately small so the (alpha, beta) sampling distribution can be
# swapped (uniform, Dirichlet, etc.) without touching either helper.
# -----------------------------------------------------------------------------
_DUAL_MODULE_TYPES = (DualLinear, DualEmbedding, DualLayerNorm)


def set_mix(model: nn.Module, alpha: float, beta: float):
    """Walk the model and assign (alpha, beta) to every Dual* submodule."""
    a = float(alpha)
    b = float(beta)
    for m in model.modules():
        if isinstance(m, _DUAL_MODULE_TYPES):
            m.alpha = a
            m.beta = b


def mask_grads(model: nn.Module, corpus: str):
    """Null the gradients on the slot NOT belonging to `corpus`, on every Dual module.

    corpus == 'shake'  -> wipe W_w.grad and b_w.grad (wiki slot frozen this iter)
    corpus == 'wiki'   -> wipe W_s.grad and b_s.grad (shake slot frozen this iter)
    """
    if corpus == 'shake':
        slot_attrs = ('W_w', 'b_w')
    elif corpus == 'wiki':
        slot_attrs = ('W_s', 'b_s')
    else:
        raise ValueError(f"unknown corpus: {corpus!r}")
    for m in model.modules():
        if isinstance(m, _DUAL_MODULE_TYPES):
            for attr in slot_attrs:
                p = getattr(m, attr, None)
                if p is not None:
                    p.grad = None


# -----------------------------------------------------------------------------
# config + GPT class. Structurally identical to model.py's GPT; only module
# types are swapped, and weight tying ties W_s and W_w across wte/lm_head.
# -----------------------------------------------------------------------------
@dataclass
class DualGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class DualCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = DualLinear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = DualLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class DualMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = DualLinear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = DualLinear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DualBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = DualLayerNorm(config.n_embd, bias=config.bias)
        self.attn = DualCausalSelfAttention(config)
        self.ln_2 = DualLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = DualMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DualGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=DualEmbedding(config.vocab_size, config.n_embd),
            wpe=DualEmbedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([DualBlock(config) for _ in range(config.n_layer)]),
            ln_f=DualLayerNorm(config.n_embd, bias=config.bias),
        ))
        # lm_head has no bias in nanoGPT; we mirror that.
        self.lm_head = DualLinear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying across BOTH slots independently. nanoGPT ties
        # transformer.wte.weight = lm_head.weight (a single shared tensor).
        # Here we replicate the tie for W_s and W_w separately so the dual
        # decomposition is preserved end-to-end.
        self.transformer.wte.W_s = self.lm_head.W_s
        self.transformer.wte.W_w = self.lm_head.W_w

        # Re-apply nanoGPT-style init to ensure tied tensors aren't double-
        # initialized in a weird order. _init_dual_weights inits W_s and then
        # copies into W_w so the two slots are byte-identical at iter 0.
        self._init_dual_weights()
        # GPT-2's scaled init applies to residual projections (c_proj.*) so
        # the variance after `n_layer` residual additions is bounded. We rescale
        # W_s and then copy into W_w so the byte-identity holds after the
        # rescale. With alpha + beta = 1 and W_s == W_w the effective weight
        # is exactly the (correctly-scaled) standard nanoGPT init at every
        # alpha in [0, 1].
        std_resid = 0.02 / math.sqrt(2 * config.n_layer)
        with torch.no_grad():
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.W_s'):
                    torch.nn.init.normal_(p, mean=0.0, std=std_resid)
            # Re-copy W_s into W_w for every Dual* module so the c_proj
            # rescale (and any earlier init) is mirrored into W_w.
            for m in self.modules():
                if isinstance(m, _DUAL_MODULE_TYPES):
                    m.W_w.data.copy_(m.W_s.data)
                    if getattr(m, 'b_w', None) is not None:
                        m.b_w.data.copy_(m.b_s.data)

        # Sanity assert: every Dual* module's W_s and W_w must be byte-identical
        # at the end of init. This is the invariant the trainer's mini-run
        # verification also checks.
        for name, m in self.named_modules():
            if isinstance(m, _DUAL_MODULE_TYPES):
                assert torch.equal(m.W_s.data, m.W_w.data), \
                    f"W_s and W_w must be identical at iter 0 ({name})"
                if getattr(m, 'b_w', None) is not None:
                    assert torch.equal(m.b_s.data, m.b_w.data), \
                        f"b_s and b_w must be identical at iter 0 ({name})"

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_dual_weights(self):
        """Reset every Dual* module's parameters to nanoGPT's init scheme, with
        W_s and W_w BYTE-IDENTICAL at iter 0.

        Init W_s with nanoGPT's scheme, then copy W_s.data into W_w.data. With
        alpha + beta = 1 the materialized weight W = alpha*W_s + (1-alpha)*W_w
        equals W_s for any alpha in [0, 1], so the iter-0 network is exactly
        the standard nanoGPT init across the entire slider range. The trainer
        then forces the two slots apart via Beta(0.5, 0.5) alpha sampling and
        per-corpus gradient routing.

        - DualLinear:     W_s ~ normal(0, 0.02); b_s = zeros; W_w := W_s; b_w := b_s.
        - DualEmbedding:  W_s ~ normal(0, 0.02);              W_w := W_s.
        - DualLayerNorm:  W_s = ones; b_s = zeros;            W_w := W_s; b_w := b_s.
        """
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, DualLinear):
                    nn.init.normal_(m.W_s, mean=0.0, std=0.02)
                    if m.b_s is not None:
                        nn.init.zeros_(m.b_s)
                    m.W_w.data.copy_(m.W_s.data)
                    if m.b_s is not None:
                        m.b_w.data.copy_(m.b_s.data)
                elif isinstance(m, DualEmbedding):
                    nn.init.normal_(m.W_s, mean=0.0, std=0.02)
                    m.W_w.data.copy_(m.W_s.data)
                elif isinstance(m, DualLayerNorm):
                    nn.init.ones_(m.W_s)
                    nn.init.ones_(m.W_w)
                    if m.b_s is not None:
                        nn.init.zeros_(m.b_s)
                        nn.init.zeros_(m.b_w)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        We count tied parameters once (matching nanoGPT's behavior; PyTorch's
        Module.parameters() already deduplicates tied params via id()).
        For non-embedding count, subtract the position embeddings.
        Note: with two slots, wpe contributes W_s and W_w both.
        """
        seen = set()
        n_params = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            n_params += p.numel()
        if non_embedding:
            n_params -= self.transformer.wpe.W_s.numel()
            n_params -= self.transformer.wpe.W_w.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Same surgery as model.py's GPT, but acting on dual slots."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # crop position-embedding slots
        self.transformer.wpe.W_s = nn.Parameter(self.transformer.wpe.W_s[:block_size])
        self.transformer.wpe.W_w = nn.Parameter(self.transformer.wpe.W_w[:block_size])
        self.transformer.wpe.num_embeddings = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW over both slots. Gradient routing per-batch is handled by
        mask_grads(); the optimizer sees both W_s and W_w in a single param group
        per decay class. Weight decay applies to 2D tensors as in nanoGPT."""
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, "
              f"with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, "
              f"with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                      **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Same arithmetic as model.py; we treat the parameter count as the
        deduped tied count, which matches GPT.get_num_params semantics."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Identical signature to GPT.generate. Uses whatever (alpha, beta) the
        Dual* modules currently hold (set via set_mix(model, alpha, beta))."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

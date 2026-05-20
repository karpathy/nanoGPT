"""GatedGPT: single transformer with per-neuron mask-based specialty routing.

The third design (after dual-slot and BTM) for the dual-source slider.

Three mask groups, each fixed at init from Beta(0.5, 0.5) (arcsine — tail-
heavy U-shape, ~20% mass near 0, ~20% near 1, the rest in middle):

  - M_embd (n_embd,):       GLOBAL per-residual-stream-channel mask.
                            Applied at: wte+wpe output, every attn c_proj
                            output, every mlp c_proj output, and ln_f
                            output before lm_head. The residual stream is
                            implicitly per-channel-gated because every
                            write to it is gated. lm_head sees only the
                            active channels at any given alpha.
  - M_mlp[i] (4*n_embd,):   per-layer MLP-inner-unit mask. Applied to the
                            post-GELU hidden activations of each block.
  - M_attn[i] (n_head,):    per-layer attention-head mask. Applied to each
                            head's output before c_proj.

During forward, a scalar alpha gates each unit by a tent/bell function
peaked at alpha=m with adaptive span:

    span(m) = max(m, 1 - m)                       # 1 for specialists, 0.5 for halfsies
    gate(unit, alpha) = cos^2(pi/2 * |alpha - m| / span(m))

For specialists (m=0 or m=1) span=1 gives a smooth ramp from 0 at the
opposite corner to 1 at the matched corner. For halfsies (m=0.5) span=0.5
gives a bell peaked at alpha=0.5 with EXACTLY zero at alpha=0 and alpha=1.
For m in between, an asymmetric bell peaks at alpha=m with exactly zero at
the further endpoint. At the extremes alpha=0 and alpha=1, ONLY the
respective specialists fire; halfsies (and all blended neurons) are
smoothly turned off.

LayerNorms stay un-gated by design: per-channel gating BEFORE LN is
partially undone by LN's mean subtraction and variance renormalization,
so we gate AFTER LN where it matters (residual-stream gating is on the
things that WRITE to the residual stream, not on LN's output that goes
into Q/K/V inside the block).

During training, alpha is sampled per iter from Beta(0.5, 0.5) (concentrating
on corners), and the corpus the batch is drawn from is chosen by Bernoulli(alpha)
— high alpha favors shake batches, low favors ts. Specialists end up trained
mostly on their own corpus, halfsies on both.

This is closer to fixed-routing Mixture-of-Experts than to weight averaging:
one model, one set of parameters, alpha modulates which subnetwork is active.
The slider exposes that routing continuously to the user.
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Mask sampling — Beta(0.5, 0.5) via inverse CDF (sin^2(pi*u/2))
# ============================================================================

def sample_beta_half_mask(shape, seed):
    """Sample a Beta(0.5, 0.5)-distributed tensor of the given shape.

    Beta(0.5, 0.5) is the arcsine distribution: f(x) = 1/(pi*sqrt(x(1-x))),
    most mass near x=0 and x=1, with a U-shaped density. We sample by inverse
    CDF transform: if U ~ Uniform[0,1] then sin^2(pi*U/2) ~ Beta(0.5, 0.5).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    u = torch.rand(shape, generator=g)
    return torch.sin(math.pi / 2 * u) ** 2


# ============================================================================
# Corpus-routed gate (straight-through estimator)
# ============================================================================
#
# Forward uses the smooth alpha-gate (so the model is exposed to mid-alpha
# behavior at training time). Backward routes gradient through the *hard*
# corpus-based gate: M for shake batches, 1-M for ts batches. Specialists
# (m=1 for shake) get full gradient from their own corpus regardless of
# alpha, and zero from the other corpus. Halfsies (m=0.5) get half from
# either. Cross-corpus leakage on specialists is plugged.
#
# At eval time (no backward, or no corpus passed), the gate is just a
# scalar multiply with the smooth alpha-gate — no autograd indirection.

def _smooth_tent(alpha, M, narrowness=1.0):
    """Tent/bell gate centered at α=m with adaptive span = max(m, 1-m) · narrowness.

    `narrowness=1.0` (default): each neuron fires across the full α∈[0,1]:
      - m=0 / m=1 (specialists): span=1, smooth ramp 0→1 from opposite to matched corner.
      - m=0.5 (halfsies): span=0.5, bell peaked at α=0.5, exactly 0 at both endpoints.

    `narrowness < 1`: each neuron's fire zone is correspondingly narrow.
      - At narrowness=0.25, m=1 fires only at α > ~0.83 (only the strong specialists
        fire near the corners); halfsies fire only in α ∈ [0.4, 0.6]; etc.
      - This produces the "more neurons at the corners" property when combined
        with the Beta(0.5,0.5) mask distribution: at α=1, only m>0.8-ish units
        fire (~30% of total per Beta(0.5,0.5)); at α=0.5, only m≈0.5 fires (~13%).

    Uses cos²(π/2 · |α-m|/span) for C^∞ smoothness; outside the support the rel
    is clamped to 1 so cos²(π/2) = 0 (exactly off).
    """
    span = torch.maximum(M, 1.0 - M) * narrowness
    rel = ((alpha - M).abs() / span).clamp(max=1.0)
    return torch.cos(math.pi * 0.5 * rel).pow(2)


class _CorpusRoutedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M, alpha, corpus, narrowness):
        # corpus: 1.0 for shake, 0.0 for ts. Stored as a Python float on ctx.
        ctx.save_for_backward(M)
        ctx.alpha = float(alpha)
        ctx.corpus = float(corpus)
        ctx.narrowness = float(narrowness)
        return x * _smooth_tent(alpha, M, narrowness)

    @staticmethod
    def backward(ctx, grad_output):
        M, = ctx.saved_tensors
        if ctx.corpus >= 0.5:
            # shake: only m near 1 get gradient
            gate_bwd = M
        else:
            # ts: only m near 0 get gradient (i.e., 1 - M near 1)
            gate_bwd = 1.0 - M
        # backward gate is independent of narrowness (it's the hard corpus mask,
        # not the smooth forward shape)
        return grad_output * gate_bwd, None, None, None, None


def gate_with_corpus(x, M, alpha, corpus, narrowness=1.0):
    """Smooth tent-gate in forward, corpus-routed gate in backward.

    If `corpus is None` (or x doesn't need grad), reduces to a plain multiply
    with the smooth tent-gate (used at eval time so we don't pay for
    autograd-function overhead).
    """
    if corpus is None or not torch.is_grad_enabled():
        return x * _smooth_tent(alpha, M, narrowness)
    return _CorpusRoutedGate.apply(x, M, alpha, corpus, narrowness)


# ============================================================================
# Model
# ============================================================================

@dataclass
class GatedGPTConfig:
    vocab_size: int = 75
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    bias: bool = False
    mask_seed: int = 1337       # fixed seed for reproducible mask sampling
    tent_narrowness: float = 1.0  # 1.0 = full-width tent; <1 narrows each neuron's fire-zone


class GatedSelfAttention(nn.Module):
    def __init__(self, config: GatedGPTConfig, layer_idx: int, M_embd: torch.Tensor):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.narrowness = config.tent_narrowness
        mask = sample_beta_half_mask((config.n_head,), seed=config.mask_seed + 100 * layer_idx + 1)
        self.register_buffer('M_head', mask)
        self.register_buffer('M_embd', M_embd)

    def forward(self, x: torch.Tensor, alpha: float, corpus) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, H, T, hd)
        y = gate_with_corpus(y, self.M_head.view(1, self.n_head, 1, 1), alpha, corpus, self.narrowness)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(y)
        out = gate_with_corpus(out, self.M_embd, alpha, corpus, self.narrowness)
        return self.resid_dropout(out)


class GatedMLP(nn.Module):
    def __init__(self, config: GatedGPTConfig, layer_idx: int, M_embd: torch.Tensor):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.narrowness = config.tent_narrowness
        mask = sample_beta_half_mask((4 * config.n_embd,), seed=config.mask_seed + 100 * layer_idx + 2)
        self.register_buffer('M_inner', mask)
        self.register_buffer('M_embd', M_embd)

    def forward(self, x: torch.Tensor, alpha: float, corpus) -> torch.Tensor:
        h = F.gelu(self.c_fc(x))                                                            # (B, T, 4D)
        h = gate_with_corpus(h, self.M_inner, alpha, corpus, self.narrowness)               # MLP-inner gate
        h = self.c_proj(h)
        h = gate_with_corpus(h, self.M_embd, alpha, corpus, self.narrowness)                # residual-stream gate
        return self.dropout(h)


class GatedBlock(nn.Module):
    def __init__(self, config: GatedGPTConfig, layer_idx: int, M_embd: torch.Tensor):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = GatedSelfAttention(config, layer_idx, M_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GatedMLP(config, layer_idx, M_embd)

    def forward(self, x: torch.Tensor, alpha: float, corpus) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), alpha, corpus)
        x = x + self.mlp(self.ln_2(x), alpha, corpus)
        return x


class GatedGPT(nn.Module):
    def __init__(self, config: GatedGPTConfig):
        super().__init__()
        self.config = config
        # Global per-channel mask for the residual-stream (n_embd) dimension.
        # Sampled once with a layer-independent seed so re-runs reproduce it.
        # Shared across every module that touches the residual stream
        # (embeddings, every block's c_proj output, and ln_f output before lm_head).
        M_embd = sample_beta_half_mask((config.n_embd,), seed=config.mask_seed)
        self.register_buffer('M_embd', M_embd)
        self.narrowness = config.tent_narrowness

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([GatedBlock(config, i, M_embd) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, alpha: float, targets=None, corpus=None):
        """Forward pass.

        `corpus` controls the backward routing: 1.0 = shake (specialists with
        m near 1 receive gradient), 0.0 = ts (specialists with m near 0
        receive gradient), None = eval / smooth gate everywhere (no routing
        autograd overhead).
        """
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # Gate the initial residual stream (embedding output)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = gate_with_corpus(x, self.M_embd, alpha, corpus, self.narrowness)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, alpha, corpus)
        # ln_f normalizes the residual stream; gate again so lm_head only sees
        # the active channels at this alpha
        x = self.transformer.ln_f(x)
        x = gate_with_corpus(x, self.M_embd, alpha, corpus, self.narrowness)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, alpha: float, max_new_tokens: int,
                 temperature: float = 1.0, top_k=None):
        # eval / generation: corpus=None so the gate is just a multiply (no autograd routing)
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, alpha, corpus=None)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def mask_summary(self):
        """Quick per-mask histogram for sanity-checking the Beta(0.5,0.5) sample."""
        def stats(M):
            return {
                'mean': float(M.mean()),
                'shake_specialist (m>0.9)': int((M > 0.9).sum()),
                'ts_specialist (m<0.1)':    int((M < 0.1).sum()),
                'halfsies (0.1..0.9)':      int(((M >= 0.1) & (M <= 0.9)).sum()),
                'total': int(M.numel()),
            }
        out = {'M_embd (global residual-stream)': stats(self.M_embd)}
        for i, block in enumerate(self.transformer.h):
            out[f'layer_{i}_attn_heads'] = stats(block.attn.M_head)
            out[f'layer_{i}_mlp_inner']  = stats(block.mlp.M_inner)
        return out


# ============================================================================
# Training
# ============================================================================

def train_gated(model, get_shake_batch, get_ts_batch, n_iters,
                lr=1e-3, warmup=100, lr_decay_iters=None, min_lr=1e-4,
                beta2=0.99, weight_decay=0.1, grad_clip=1.0,
                alpha_dist='beta_half',
                log_interval=100, eval_interval=500, eval_iters=200,
                get_shake_val=None, get_ts_val=None,
                device='cuda', amp_dtype=torch.bfloat16):
    """Train GatedGPT.

    Each iter:
      1. Sample alpha (Beta(0.5,0.5) by default; 'uniform' also supported).
      2. Choose corpus by Bernoulli(alpha): shake with prob alpha, else ts.
      3. Forward through the gated model with this alpha.
      4. CE loss, backward, AdamW step.

    Specialists end up trained mostly on their own corpus (the gate keeps the
    off-specialty unit's gradient at near zero when alpha is at the off-corner,
    and the corpus matches the gate), and halfsies are trained on both.
    """
    if lr_decay_iters is None:
        lr_decay_iters = n_iters

    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': weight_decay},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, beta2), fused=(device == 'cuda'),
    )

    def get_lr(it):
        if it < warmup:
            return lr * (it + 1) / warmup
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup) / (lr_decay_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def sample_alpha() -> float:
        if alpha_dist == 'beta_half':
            u = torch.rand(1).item()
            return math.sin(math.pi / 2 * u) ** 2
        return torch.rand(1).item()

    @torch.no_grad()
    def estimate_val(get_batch_fn, alpha):
        if get_batch_fn is None:
            return None
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn()
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(X, alpha, Y, corpus=None)  # eval: smooth gate, no routing
            losses[k] = loss.item()
        model.train()
        return losses.mean().item()

    # Counters for diagnostics
    n_shake = 0
    n_ts = 0

    model.train()
    t_start = time.time()
    t_log = t_start
    for it in range(n_iters):
        cur_lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        alpha = sample_alpha()
        use_shake = torch.rand(1).item() < alpha
        if use_shake:
            X, Y = get_shake_batch(); n_shake += 1
        else:
            X, Y = get_ts_batch();    n_ts += 1

        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            _, loss = model(X, alpha, Y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (it + 1) % log_interval == 0:
            dt = time.time() - t_log
            print(f"iter {it+1:5d} | alpha {alpha:.2f} | {'shake' if use_shake else '   ts'} | "
                  f"loss {loss.item():.4f} | lr {cur_lr:.6f} | dt {dt:.1f}s", flush=True)
            t_log = time.time()

        if (it + 1) % eval_interval == 0:
            v_sh_1 = estimate_val(get_shake_val, alpha=1.0)
            v_ts_0 = estimate_val(get_ts_val,    alpha=0.0)
            v_sh_5 = estimate_val(get_shake_val, alpha=0.5)
            v_ts_5 = estimate_val(get_ts_val,    alpha=0.5)
            print(f"  >>> step {it+1}: shake@a=1.0={v_sh_1:.3f}  ts@a=0.0={v_ts_0:.3f}  "
                  f"shake@a=0.5={v_sh_5:.3f}  ts@a=0.5={v_ts_5:.3f}  "
                  f"corpus split so far: {n_shake} shake / {n_ts} ts", flush=True)

    print(f"total training time: {time.time() - t_start:.1f}s  "
          f"final split: {n_shake} shake / {n_ts} ts", flush=True)
    return model

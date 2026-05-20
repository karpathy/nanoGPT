"""Branch-Train-Merge utilities for the dual-source slider demo.

The cleaner alternative to dual-slot training: train two standard nanoGPT-style
models from a byte-identical shared init, one per corpus, with no alpha-mix
and no mask_grads. At inference, merge them via either naive linear
interpolation or Git Re-Basin permutation-aligned interpolation, exposed
to the user as two side-by-side sliders.

Usage:
    from btm_lib import GPT, GPTConfig, train_one_corpus, merge_naive, merge_rebasin

    config = GPTConfig(vocab_size=75, n_layer=6, n_head=6, n_embd=384,
                       block_size=256, dropout=0.2)

    # Build A. Snapshot the init. Build B from the same init.
    model_a = GPT(config).to(device)
    init_sd = {k: v.clone() for k, v in model_a.state_dict().items()}
    model_b = GPT(config).to(device)
    model_b.load_state_dict(init_sd)

    # Train each on its own corpus.
    train_one_corpus(model_a, shake_get_batch, n_iters=5000, ...)
    train_one_corpus(model_b, ts_get_batch,    n_iters=5000, ...)

    # Merge.
    sd_merged = merge_naive(model_a.state_dict(), model_b.state_dict(), alpha=0.5)
    # or:
    sd_merged = merge_rebasin(model_a.state_dict(), model_b.state_dict(),
                              alpha=0.5, n_layer=6)
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model — minimal nanoGPT-style GPT, char-level, weight-tied lm_head
# ============================================================================

@dataclass
class GPTConfig:
    vocab_size: int = 75
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    bias: bool = False  # no bias anywhere; matches nanoGPT char-shake


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Training — Karpathy nanoGPT char-shake recipe, single corpus, single model
# ============================================================================

def train_one_corpus(model, get_batch_fn, n_iters,
                     lr=1e-3, warmup=100, lr_decay_iters=None, min_lr=1e-4,
                     beta2=0.99, weight_decay=0.1, grad_clip=1.0,
                     log_interval=100, eval_interval=500, eval_iters=200,
                     get_val_batch_fn=None,
                     device='cuda', amp_dtype=torch.bfloat16):
    """Train a single GPT model on a single corpus, exactly Karpathy-style.

    No alpha-mix, no mask_grads, no dual-slot anything. This is the standard
    nanoGPT char-shake training loop applied to one corpus. Each iter is a
    full-rate optimizer step on a single-corpus batch.
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

    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def get_lr(it):
        if it < warmup:
            return lr * (it + 1) / warmup
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup) / (lr_decay_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    @torch.no_grad()
    def estimate_val():
        if get_val_batch_fn is None:
            return None
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_val_batch_fn()
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        model.train()
        return losses.mean().item()

    model.train()
    t_start = time.time()
    t_log = t_start
    for it in range(n_iters):
        cur_lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        X, Y = get_batch_fn()
        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            _, loss = model(X, Y)
        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (it + 1) % log_interval == 0:
            dt = time.time() - t_log
            print(f"iter {it+1:5d} | loss {loss.item():.4f} | lr {cur_lr:.6f} | dt {dt:.1f}s", flush=True)
            t_log = time.time()

        if (it + 1) % eval_interval == 0:
            val = estimate_val()
            if val is not None:
                print(f"  >>> step {it+1}: val loss {val:.4f}", flush=True)

    print(f"total training time: {time.time() - t_start:.1f}s", flush=True)
    return model


# ============================================================================
# Merge — naive linear interp, and Git Re-Basin permutation-aligned interp
# ============================================================================

def merge_naive(sd_a, sd_b, alpha):
    """W_mix = alpha * sd_a + (1 - alpha) * sd_b, parameter-wise.

    The full-rank cousin of model-soup / task-arithmetic merging.
    """
    return {k: alpha * sd_a[k] + (1.0 - alpha) * sd_b[k] for k in sd_a}


def align_rebasin_mlp(sd_a, sd_b, n_layer):
    """Permute sd_b's MLP inner-dim units per layer to align with sd_a's.

    This is the MLP-only version of Git Re-Basin (Ainsworth et al. 2022).
    Each transformer block has a 4*n_embd-wide MLP hidden dim that is purely
    local to the block: c_fc maps n_embd -> 4*n_embd and c_proj maps back, so
    you can permute that inner axis freely as long as you apply the same
    permutation to c_fc's rows and c_proj's columns. The residual-stream
    dimension is NOT permuted (it propagates across layers via the residual
    adds, which would require a globally consistent permutation that is much
    more involved). Attention heads are also not permuted in this version —
    head-block permutation across c_attn(Q,K,V) and c_proj is doable but
    requires more careful handling and is left as a TODO.

    The cost matrix for each layer is -W_fc_a @ W_fc_b.T (negated so
    scipy.optimize.linear_sum_assignment, which minimizes, maximizes the
    inner-product alignment).
    """
    from scipy.optimize import linear_sum_assignment

    sd_b_aligned = {k: v.clone() for k, v in sd_b.items()}

    for i in range(n_layer):
        key_fc_w = f'transformer.h.{i}.mlp.c_fc.weight'
        key_fc_b = f'transformer.h.{i}.mlp.c_fc.bias'
        key_proj_w = f'transformer.h.{i}.mlp.c_proj.weight'

        W_fc_a = sd_a[key_fc_w].float()                    # (4D, D)
        W_fc_b = sd_b[key_fc_w].float()                    # (4D, D)

        cost = -(W_fc_a @ W_fc_b.T).cpu().numpy()          # (4D, 4D)
        _, col_ind = linear_sum_assignment(cost)
        perm = torch.tensor(col_ind, dtype=torch.long, device=W_fc_b.device)

        # c_fc rows: permute
        sd_b_aligned[key_fc_w] = sd_b[key_fc_w][perm]
        if key_fc_b in sd_b:
            sd_b_aligned[key_fc_b] = sd_b[key_fc_b][perm]
        # c_proj columns (input axis): permute
        sd_b_aligned[key_proj_w] = sd_b[key_proj_w][:, perm]
        # c_proj output bias (if present) is per-residual-dim, NOT permuted

    return sd_b_aligned


def merge_rebasin(sd_a, sd_b, alpha, n_layer):
    """Align sd_b's MLP units to sd_a's, then linearly interpolate."""
    sd_b_aligned = align_rebasin_mlp(sd_a, sd_b, n_layer)
    return merge_naive(sd_a, sd_b_aligned, alpha)


# ============================================================================
# Self-test
# ============================================================================

def _self_test():
    """Quick sanity check that the merge functions behave correctly at endpoints."""
    cfg = GPTConfig(vocab_size=10, n_layer=2, n_head=2, n_embd=16, block_size=8, dropout=0.0)
    torch.manual_seed(0)
    m_a = GPT(cfg)
    init_sd = {k: v.clone() for k, v in m_a.state_dict().items()}
    m_b = GPT(cfg)
    m_b.load_state_dict(init_sd)
    # Byte-identical init
    for (na, pa), (nb, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
        assert torch.allclose(pa, pb), f"init mismatch at {na}"
    # Perturb m_b so they differ
    with torch.no_grad():
        for p in m_b.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    sd_a = m_a.state_dict()
    sd_b = m_b.state_dict()

    # alpha=1 -> sd_a; alpha=0 -> sd_b
    s = merge_naive(sd_a, sd_b, alpha=1.0)
    for k in sd_a:
        assert torch.allclose(s[k], sd_a[k]), f"naive alpha=1 mismatch at {k}"
    s = merge_naive(sd_a, sd_b, alpha=0.0)
    for k in sd_b:
        assert torch.allclose(s[k], sd_b[k]), f"naive alpha=0 mismatch at {k}"

    # Re-Basin: alpha=1 still returns sd_a; alpha=0 returns aligned sd_b (not raw sd_b)
    s = merge_rebasin(sd_a, sd_b, alpha=1.0, n_layer=cfg.n_layer)
    for k in sd_a:
        assert torch.allclose(s[k], sd_a[k]), f"rebasin alpha=1 mismatch at {k}"

    print("btm_lib self-test passed")


if __name__ == '__main__':
    _self_test()

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
# Full Re-Basin: every permutable component of the transformer
# ============================================================================
#
# Three permutation groups exist in this architecture:
#
#   1. P_embd: a single global permutation of the n_embd residual-stream
#      dimension. Affects wte cols, wpe cols, every ln_*'s per-element gain,
#      every c_attn's input cols, every attention c_proj's output rows,
#      every mlp.c_fc's input cols, every mlp.c_proj's output rows, ln_f's
#      gain, and lm_head cols (tied to wte). Coupled across the whole
#      network — has to be ONE permutation, not per-layer.
#
#   2. P_mlp[i]: per-layer permutation of the 4*n_embd MLP inner dim.
#      Affects layer i's mlp.c_fc rows and mlp.c_proj cols. Layer-local.
#
#   3. P_head[i]: per-layer permutation of the n_head attention heads
#      (permutes head-blocks of head_dim each within Q, K, V row sections
#      of c_attn, and within input cols of attention c_proj). Layer-local.
#      Q/K/V must receive the SAME head permutation because head h's query
#      attends to head h's key.
#
# Coordinate descent: initialize all perms to identity, then iterate:
#   - solve P_embd LSAP holding all layer-local perms fixed
#   - solve each P_mlp[i] LSAP holding P_embd and P_head fixed
#   - solve each P_head[i] LSAP holding P_embd and P_mlp fixed
#   - repeat until no permutation changes (or max_iters reached)
#
# We do NOT permute the within-head-dim symmetry (you can jointly permute
# Q's and K's head_dim rows for a single head and Q@K^T is invariant) —
# that's a more exotic symmetry rarely load-bearing in practice.


def _attn_head_row_idx(P_head_layer, n_head, head_dim, n_embd, device):
    """Build the row index that permutes the head-blocks of c_attn.
    c_attn rows: [Q rows (0:D), K rows (D:2D), V rows (2D:3D)], each split
    into n_head head-blocks of head_dim each. We permute the head-blocks
    using P_head_layer (the SAME permutation across Q, K, V)."""
    parts = []
    for block_offset in (0, n_embd, 2 * n_embd):
        for h in P_head_layer.tolist():
            parts.extend(range(block_offset + h * head_dim, block_offset + (h + 1) * head_dim))
    return torch.tensor(parts, dtype=torch.long, device=device)


def _attn_head_col_idx(P_head_layer, n_head, head_dim, device):
    """Build the col index that permutes head-blocks of attention c_proj's input axis."""
    parts = []
    for h in P_head_layer.tolist():
        parts.extend(range(h * head_dim, (h + 1) * head_dim))
    return torch.tensor(parts, dtype=torch.long, device=device)


def _apply_all_perms(sd, P_embd, P_mlp, P_head, n_layer, n_head, n_embd, device):
    """Return a new state_dict with P_embd, P_mlp[i], P_head[i] applied to sd_b.
    Assumes the btm_lib.GPT architecture: bias=False everywhere, lm_head tied to wte."""
    head_dim = n_embd // n_head
    out = {}

    # Embeddings: wte (V, D) and wpe (T, D) — permute cols by P_embd
    wte = sd['transformer.wte.weight'][:, P_embd]
    out['transformer.wte.weight'] = wte
    if 'lm_head.weight' in sd:
        out['lm_head.weight'] = wte  # tied
    out['transformer.wpe.weight'] = sd['transformer.wpe.weight'][:, P_embd]
    # ln_f: per-element gain, permute by P_embd
    out['transformer.ln_f.weight'] = sd['transformer.ln_f.weight'][P_embd]

    for i in range(n_layer):
        # LayerNorms
        out[f'transformer.h.{i}.ln_1.weight'] = sd[f'transformer.h.{i}.ln_1.weight'][P_embd]
        out[f'transformer.h.{i}.ln_2.weight'] = sd[f'transformer.h.{i}.ln_2.weight'][P_embd]

        # c_attn (3D, D): cols by P_embd, rows by head-block permutation
        c_attn = sd[f'transformer.h.{i}.attn.c_attn.weight'][:, P_embd]
        c_attn = c_attn[_attn_head_row_idx(P_head[i], n_head, head_dim, n_embd, device)]
        out[f'transformer.h.{i}.attn.c_attn.weight'] = c_attn

        # attn c_proj (D, D): rows by P_embd, cols by head-block permutation
        c_proj = sd[f'transformer.h.{i}.attn.c_proj.weight'][P_embd]
        c_proj = c_proj[:, _attn_head_col_idx(P_head[i], n_head, head_dim, device)]
        out[f'transformer.h.{i}.attn.c_proj.weight'] = c_proj

        # mlp c_fc (4D, D): cols by P_embd, rows by P_mlp[i]
        c_fc = sd[f'transformer.h.{i}.mlp.c_fc.weight'][:, P_embd]
        c_fc = c_fc[P_mlp[i]]
        out[f'transformer.h.{i}.mlp.c_fc.weight'] = c_fc

        # mlp c_proj (D, 4D): rows by P_embd, cols by P_mlp[i]
        c_proj_mlp = sd[f'transformer.h.{i}.mlp.c_proj.weight'][P_embd]
        c_proj_mlp = c_proj_mlp[:, P_mlp[i]]
        out[f'transformer.h.{i}.mlp.c_proj.weight'] = c_proj_mlp

    return out


def align_rebasin_full(sd_a, sd_b, n_layer, n_head, n_embd, max_iters=20, verbose=False):
    """Full Git Re-Basin alignment over every permutable component.

    Implements coordinate descent over the three permutation groups (P_embd,
    P_mlp per layer, P_head per layer) as described in Ainsworth et al. 2022,
    adapted to the GPT architecture in btm_lib.GPT. Each pass solves an LSAP
    per group holding the others fixed. Returns sd_b with all permutations
    applied so that it aligns with sd_a; you can then merge_naive(sd_a, this).

    Compute: per pass is one D-sized LSAP plus n_layer (4D)-sized LSAPs plus
    n_layer H-sized LSAPs, where D=n_embd. The 4D LSAP is the bottleneck
    (~few seconds on CPU). Typically converges in 2-8 passes for shared-init
    slot pairs.
    """
    from scipy.optimize import linear_sum_assignment

    head_dim = n_embd // n_head
    device = sd_a['transformer.wte.weight'].device

    # Initialize all permutations to identity
    P_embd = torch.arange(n_embd, device=device)
    P_mlp = [torch.arange(4 * n_embd, device=device) for _ in range(n_layer)]
    P_head = [torch.arange(n_head, device=device) for _ in range(n_layer)]

    def lsap(cost):
        """Solve assignment problem (minimize sum of cost). Returns col_ind as torch."""
        _, col_ind = linear_sum_assignment(cost.cpu().numpy())
        return torch.tensor(col_ind, dtype=torch.long, device=device)

    for pass_i in range(max_iters):
        old_P_embd = P_embd.clone()
        old_P_mlp = [p.clone() for p in P_mlp]
        old_P_head = [p.clone() for p in P_head]

        # ------------------------------------------------------------------
        # Update P_embd: cost over every tensor where P_embd is one of the axes.
        # Build B with current head and mlp perms applied, but NOT P_embd, then
        # compute inner-product cost matrix along the D axis.
        # ------------------------------------------------------------------
        cost_embd = torch.zeros(n_embd, n_embd, device=device, dtype=torch.float32)

        # wte (V, D): inner product along V axis -> (D, D)
        cost_embd += sd_a['transformer.wte.weight'].float().T @ sd_b['transformer.wte.weight'].float()
        # wpe (T, D): same
        cost_embd += sd_a['transformer.wpe.weight'].float().T @ sd_b['transformer.wpe.weight'].float()
        # ln_f (D,): outer product
        cost_embd += sd_a['transformer.ln_f.weight'].float().unsqueeze(1) @ sd_b['transformer.ln_f.weight'].float().unsqueeze(0)

        for i in range(n_layer):
            # ln_1, ln_2 — per-element gain, outer product
            cost_embd += sd_a[f'transformer.h.{i}.ln_1.weight'].float().unsqueeze(1) @ sd_b[f'transformer.h.{i}.ln_1.weight'].float().unsqueeze(0)
            cost_embd += sd_a[f'transformer.h.{i}.ln_2.weight'].float().unsqueeze(1) @ sd_b[f'transformer.h.{i}.ln_2.weight'].float().unsqueeze(0)

            # c_attn (3D, D) — D is INPUT axis (cols); rows already have current P_head applied
            row_idx = _attn_head_row_idx(P_head[i], n_head, head_dim, n_embd, device)
            c_attn_a = sd_a[f'transformer.h.{i}.attn.c_attn.weight'].float()
            c_attn_b_h = sd_b[f'transformer.h.{i}.attn.c_attn.weight'][row_idx].float()
            cost_embd += c_attn_a.T @ c_attn_b_h

            # attn c_proj (D, D) — D is OUTPUT axis (rows); cols have current P_head applied
            col_idx = _attn_head_col_idx(P_head[i], n_head, head_dim, device)
            c_proj_a = sd_a[f'transformer.h.{i}.attn.c_proj.weight'].float()
            c_proj_b_h = sd_b[f'transformer.h.{i}.attn.c_proj.weight'][:, col_idx].float()
            cost_embd += c_proj_a @ c_proj_b_h.T

            # mlp c_fc (4D, D) — D is INPUT axis (cols); rows have current P_mlp[i] applied
            c_fc_a = sd_a[f'transformer.h.{i}.mlp.c_fc.weight'].float()
            c_fc_b_m = sd_b[f'transformer.h.{i}.mlp.c_fc.weight'][P_mlp[i]].float()
            cost_embd += c_fc_a.T @ c_fc_b_m

            # mlp c_proj (D, 4D) — D is OUTPUT axis (rows); cols have current P_mlp[i] applied
            c_proj_mlp_a = sd_a[f'transformer.h.{i}.mlp.c_proj.weight'].float()
            c_proj_mlp_b_m = sd_b[f'transformer.h.{i}.mlp.c_proj.weight'][:, P_mlp[i]].float()
            cost_embd += c_proj_mlp_a @ c_proj_mlp_b_m.T

        P_embd = lsap(-cost_embd)

        # ------------------------------------------------------------------
        # Update each P_mlp[i]: cost is across mlp.c_fc rows and mlp.c_proj cols
        # with current P_embd applied to their D-axes.
        # ------------------------------------------------------------------
        for i in range(n_layer):
            c_fc_a = sd_a[f'transformer.h.{i}.mlp.c_fc.weight'].float()                   # (4D, D)
            c_fc_b = sd_b[f'transformer.h.{i}.mlp.c_fc.weight'][:, P_embd].float()        # (4D, D)
            c_proj_a = sd_a[f'transformer.h.{i}.mlp.c_proj.weight'].float()               # (D, 4D)
            c_proj_b = sd_b[f'transformer.h.{i}.mlp.c_proj.weight'][P_embd].float()       # (D, 4D)

            cost = c_fc_a @ c_fc_b.T + c_proj_a.T @ c_proj_b                              # (4D, 4D)
            P_mlp[i] = lsap(-cost)

        # ------------------------------------------------------------------
        # Update each P_head[i]: cost is across c_attn rows (Q/K/V head blocks)
        # and attn c_proj cols (head blocks) with current P_embd applied.
        # ------------------------------------------------------------------
        for i in range(n_layer):
            c_attn_a = sd_a[f'transformer.h.{i}.attn.c_attn.weight'].float()                # (3D, D)
            c_attn_b = sd_b[f'transformer.h.{i}.attn.c_attn.weight'][:, P_embd].float()     # (3D, D)
            c_proj_a = sd_a[f'transformer.h.{i}.attn.c_proj.weight'].float()                # (D, D)
            c_proj_b = sd_b[f'transformer.h.{i}.attn.c_proj.weight'][P_embd].float()        # (D, D)

            # Reshape c_attn rows as (3, H, hd, D) so we can pair heads across A and B
            a_r = c_attn_a.view(3, n_head, head_dim, n_embd).permute(1, 0, 2, 3).reshape(n_head, -1)  # (H, 3*hd*D)
            b_r = c_attn_b.view(3, n_head, head_dim, n_embd).permute(1, 0, 2, 3).reshape(n_head, -1)
            cost_head_attn = a_r @ b_r.T                                                            # (H, H)

            # Reshape c_proj cols as (D, H, hd) and pair heads
            a_p = c_proj_a.view(n_embd, n_head, head_dim).permute(1, 0, 2).reshape(n_head, -1)      # (H, D*hd)
            b_p = c_proj_b.view(n_embd, n_head, head_dim).permute(1, 0, 2).reshape(n_head, -1)
            cost_head_proj = a_p @ b_p.T                                                            # (H, H)

            P_head[i] = lsap(-(cost_head_attn + cost_head_proj))

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        changed_embd = not torch.equal(P_embd, old_P_embd)
        changed_mlp = sum(not torch.equal(P_mlp[i], old_P_mlp[i]) for i in range(n_layer))
        changed_head = sum(not torch.equal(P_head[i], old_P_head[i]) for i in range(n_layer))
        if verbose:
            print(f'  rebasin pass {pass_i + 1}: '
                  f'embd={"X" if changed_embd else "."}  '
                  f'mlp={changed_mlp}/{n_layer}  '
                  f'head={changed_head}/{n_layer}', flush=True)
        if not changed_embd and changed_mlp == 0 and changed_head == 0:
            if verbose:
                print(f'  converged after {pass_i + 1} pass(es)', flush=True)
            break

    return _apply_all_perms(sd_b, P_embd, P_mlp, P_head, n_layer, n_head, n_embd, device)


def merge_rebasin_full(sd_a, sd_b, alpha, n_layer, n_head, n_embd, max_iters=20, verbose=False):
    """Full Re-Basin alignment then linear interpolation."""
    sd_b_aligned = align_rebasin_full(sd_a, sd_b, n_layer, n_head, n_embd,
                                      max_iters=max_iters, verbose=verbose)
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

    # Full Re-Basin recoverability: apply a KNOWN permutation to sd_a to produce sd_perm,
    # then align_rebasin_full(sd_a, sd_perm) should recover sd_a (or something with
    # the same function — at minimum, the inner products should max out).
    D, H, hd, L = cfg.n_embd, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_layer
    P_embd_true = torch.randperm(D)
    P_mlp_true = [torch.randperm(4 * D) for _ in range(L)]
    P_head_true = [torch.randperm(H) for _ in range(L)]
    sd_perm = _apply_all_perms(sd_a, P_embd_true, P_mlp_true, P_head_true, L, H, D, sd_a['transformer.wte.weight'].device)

    # Align sd_perm back to sd_a — if we found the right inverse permutations, we recover sd_a
    sd_recovered = align_rebasin_full(sd_a, sd_perm, n_layer=L, n_head=H, n_embd=D, max_iters=30, verbose=False)
    # Validate by checking each tensor matches sd_a (up to numerical noise)
    max_err = 0.0
    for k in sd_a:
        if 'lm_head' in k:
            continue  # tied; checked via wte
        diff = (sd_recovered[k].float() - sd_a[k].float()).abs().max().item()
        max_err = max(max_err, diff)
    print(f"full re-basin recoverability test: max element-wise error = {max_err:.6e}")
    assert max_err < 1e-3, f"full re-basin failed to recover known permutation, max_err={max_err}"

    print("btm_lib self-test passed (incl. full re-basin recoverability)")


if __name__ == '__main__':
    _self_test()

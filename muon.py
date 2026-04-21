"""
Muon optimizer - MomentUm Orthogonalized by Newton-schulz

Ported from https://github.com/KellerJordan/Muon for single-GPU usage.
See: https://kellerjordan.github.io/posts/muon/

Muon should only be used for hidden 2D weight layers. Embeddings, output heads,
biases, and layernorms should be optimized with a standard method like AdamW.
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    We use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad, momentum_buf, beta=0.95, ns_steps=5, nesterov=True):
    """Compute the Muon update: momentum + Newton-Schulz orthogonalization."""
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf
    if update.ndim == 4:  # conv filters: flatten last 3 dims
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(0) / update.size(1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Single-device Muon optimizer for hidden 2D weight matrices.

    Args:
        params: iterable of parameters (should be 2D hidden weights only)
        lr: learning rate (default: 0.02)
        momentum: momentum coefficient (default: 0.95)
        weight_decay: AdamW-style weight decay (default: 0.0)
        ns_steps: number of Newton-Schulz iterations (default: 5)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                update = muon_update(
                    p.grad, state['momentum_buffer'],
                    beta=group['momentum'], ns_steps=group['ns_steps']
                )
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.reshape(p.shape), alpha=-group['lr'])

        return loss



class LowRankMuon(Muon):
 
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,ns_steps=5, rank=8, use_adaptive_rank=False):
        self.rank = rank
        self.use_adaptive_rank = use_adaptive_rank
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, rank=rank, use_adaptive_rank=use_adaptive_rank)
        super().__init__(params, defaults)

    def generate_gaussian_sketch(self, param):
        m,n = param.shape
        sketch = torch.randn(n, self.rank, device=param.device, dtype=torch.bfloat16)/n**0.5
        return sketch
    
    def low_rank_approximation(self, param):
        m,n = param.shape
        transposed = False
        if m>n:
            param = param.T
            transposed = True
        sketch = self.generate_gaussian_sketch(param)
        Y = param @ sketch
        return Y, sketch, transposed

    def muon_update(self, grad, momentum_buf, beta=0.95, ns_steps=5, nesterov=True):
        # if use_adaptive_rank:
        #     rank = min(grad.shape[0]//4, 64)  
        # else:
        rank = self.rank  
        momentum_buf.lerp_(grad, 1 - beta)
        update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf
        if update.ndim == 4:  # conv filters: flatten last 3 dims
            update = update.view(len(update), -1)
        
        update = update.bfloat16()  
        low_rank_update, sketch, transposed = self.low_rank_approximation(update)
        low_rank_update = zeropower_via_newtonschulz5(low_rank_update, steps=ns_steps)
        # low_rank_update *= max(1, low_rank_update.size(0) / low_rank_update.size(1)) ** 0.5
        out = low_rank_update @ sketch.T if not transposed else (low_rank_update @ sketch.T).T
        m, n = out.shape
        out *= max(1, m/n) ** 0.5
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                update = self.muon_update(
                    p.grad, state['momentum_buffer'],
                    beta=group['momentum'], ns_steps=group['ns_steps']
                )
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.reshape(p.shape), alpha=-group['lr'])

        return loss



class InfrequentMuon(torch.optim.Optimizer):
    """
    Muon variant that runs the expensive Newton-Schulz orthogonalization only
    every `update_freq` steps, reusing the cached update in between.
 
    Args:
        params: iterable of parameters (should be 2D hidden weights only)
        lr: learning rate (default: 0.02)
        momentum: momentum coefficient (default: 0.95)
        weight_decay: AdamW-style weight decay (default: 0.0)
        ns_steps: number of Newton-Schulz iterations (default: 5)
        update_freq: how often to recompute the NS orthogonalization (default: 5)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5, update_freq=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, update_freq=update_freq)
        super().__init__(params, defaults)
 
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
 
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['cached_update'] = torch.zeros_like(p)
                    state['step'] = 0
 
                state['step'] += 1
                beta = group['momentum']
                momentum_buf = state['momentum_buffer']
 
                # Update momentum buffer in-place, then build nesterov estimate
                # from a clone so we don't corrupt the buffer or p.grad
                momentum_buf.lerp_(p.grad, 1 - beta)
                base_update = p.grad.clone().lerp_(momentum_buf, beta)
 
                # Run Newton-Schulz on step 1, then every update_freq steps after
                if state['step'] == 1 or state['step'] % group['update_freq'] == 0:
                    shape = base_update.shape
                    flat = base_update.view(len(base_update), -1) if base_update.ndim == 4 else base_update
 
                    assert flat.ndim == 2, f"Expected 2D tensor for NS, got shape {flat.shape}"
 
                    ns_update = zeropower_via_newtonschulz5(flat, steps=group['ns_steps'])
                    ns_update *= max(1, ns_update.size(0) / ns_update.size(1)) ** 0.5
                    state['cached_update'].copy_(ns_update.reshape(shape))
 
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(state['cached_update'], alpha=-group['lr'])
 
        return loss

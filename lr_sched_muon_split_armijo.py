# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import math
import random
import torch
import numpy as np
import torch.distributed as dist
# from absl import logging
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from muon import zeropower_via_newtonschulz5

# Fix random seeds for reproducibility
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LineSearchScheduler():
    def __init__(self, optimizer, start_lr, model_paras, num_search=16, optimizer_type="SGD", injection=True, search_mode="backtrack", warmup_length=100, num_perturb_samples=3, rho=0.001, controlled_group_indices=None):
        """
        num_search: maximum number of searches
        start_lr: maximum LR to start if backtrack/ minimum LR to start if forward
        optimizer_type: Option: SGD, SGD_momentum, Adam
        """



        self.optimizer = optimizer
        self.num_search = num_search
        self.start_lr = start_lr
        # self.model = model
        self.optimizer_type = optimizer_type
        self.injection=injection
        self.prev_fvals = deque(maxlen=2)
        self.line_search_alpha = start_lr
        self.line_search_magnitude = start_lr
        self.prev_magnitude = start_lr
        self.magnitude = start_lr
        self.warmup_length = warmup_length
        self.prev_alpha = start_lr
        self.search_mode = search_mode
        self.K = num_perturb_samples
        self.rho = rho
        self.controlled_group_indices = list(range(len(self.optimizer.param_groups))) if controlled_group_indices is None else list(controlled_group_indices)
        self.controlled_group_index_set = set(self.controlled_group_indices)
        self.param_to_group = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.param_to_group[id(p)] = group
        for group_idx in self.controlled_group_indices:
            self.optimizer.param_groups[group_idx]["lr"] = self.start_lr
        self.paras = model_paras
        # self.injection_distribution = self._generate_long_tail_distribution()
        self.rule = self.get_potential_update_direction()

    
    
    def _generate_long_tail_distribution(self):
        distribution = [1.0] * 80 + [random.uniform(1.0, 2.0) for _ in range(20)]
        random.shuffle(distribution)
        return distribution
    
    def state_dict(self):
        return {
            'last_lr': self.prev_alpha,
            'line_search_alpha': self.line_search_alpha,
            'line_search_magnitude': self.line_search_magnitude,
            'prev_magnitude': self.prev_magnitude,
            'magnitude': self.magnitude,
        }
    
    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected dict, got {type(state_dict)}")

        self.prev_alpha = state_dict.get('last_lr', self.start_lr)
        self.line_search_alpha = state_dict.get('line_search_alpha', self.start_lr)
        self.line_search_magnitude = state_dict.get('line_search_magnitude', self.start_lr)
        self.prev_magnitude = state_dict.get('prev_magnitude', self.start_lr)
        self.magnitude = state_dict.get('magnitude', self.start_lr)

    def get_potential_update_direction(self, fallback_to_neg_grad=False):
        if self.optimizer_type == "AdamW":
            return self.get_potential_adam_update_direction(fallback_to_neg_grad)
        elif self.optimizer_type == "Muon":
            return self.get_potential_muon_update_direction(fallback_to_neg_grad)
        elif self.optimizer_type == "SGD_momentum":
            return self.get_potential_sgd_momentum_update_direction(fallback_to_neg_grad)
        elif self.optimizer_type == "SGD":
            return self.get_potential_sgd_update_direction(fallback_to_neg_grad)
        else:
            raise ValueError(f"Unknown optimizer_type {self.optimizer_type}")
        

    def get_potential_sgd_update_direction(self, fallback_to_neg_grad=True):
        def rule(p):
            g = p.grad
            if g is None:
                return torch.zeros_like(p)
            return -g
        return rule

    def get_potential_sgd_momentum_update_direction(self, fallback_to_neg_grad=True):
        pg0 = self.optimizer.param_groups[0]
        momentum = pg0.get("momentum", 0.0)

        def rule(p):
            g = p.grad
            if g is None:
                return torch.zeros_like(p)

            st = self.optimizer.state.get(p, {})
            if "momentum_buffer" in st:
                v = st["momentum_buffer"]
                return -(momentum * v + g)
            else:
                if fallback_to_neg_grad:
                    return -g
                else:
                    return torch.zeros_like(p)

        return rule

    def _is_muon_group(self, group):
        if "use_muon" in group:
            return bool(group["use_muon"])
        return False

    def _muon_group_indices(self):
        return self.controlled_group_indices

    def _adam_group_indices(self):
        return [
            idx for idx, group in enumerate(self.optimizer.param_groups)
            if idx not in self.controlled_group_index_set and not self._is_muon_group(group)
        ]

    def _optimizer_params(self):
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
        return params

    def _adam_direction_for_param(self, p, group, fallback_to_neg_grad=False):
        g = p.grad
        if g is None:
            return torch.zeros_like(p)

        if fallback_to_neg_grad:
            return -g

        eps = group.get("eps", 1e-8)
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        st = self.optimizer.state.get(p, {})
        if (
            "exp_avg" in st
            and "exp_avg_sq" in st
            and st.get("step", 0) > 0
        ):
            m = st["exp_avg"]
            v = st["exp_avg_sq"]
            t = st["step"] + 1
            m_new = beta1 * m + (1 - beta1) * g
            v_new = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m_new / (1 - beta1 ** t)
            v_hat = v_new / (1 - beta2 ** t)
            return -m_hat / (v_hat.sqrt() + eps)

        return -g / (g.abs() + eps)
    

    def get_potential_muon_update_direction(self, fallback_to_neg_grad=True):
        def rule(p):
            g = p.grad
            if g is None:
                return torch.zeros_like(p)

            group = self.param_to_group.get(id(p), {})
            if not self._is_muon_group(group):
                return self._adam_direction_for_param(p, group, fallback_to_neg_grad)

            if p.ndim < 2:
                return -g if fallback_to_neg_grad else torch.zeros_like(p)

            momentum = group.get("momentum", 0.95)
            st = self.optimizer.state.get(p, {})
            buf = st["momentum_buffer"] if "momentum_buffer" in st else torch.zeros_like(p)
            next_buf = torch.lerp(buf, g, 1 - momentum)
            update = torch.lerp(g, next_buf, momentum)
            if update.ndim == 4:
                update = update.view(len(update), -1)
            update = zeropower_via_newtonschulz5(update, steps=5)
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
            return -update.reshape(p.shape)

        return rule

    def get_potential_adam_update_direction(self, fallback_to_neg_grad=False):
        pg0 = self.optimizer.param_groups[self.controlled_group_indices[0]]
        eps = pg0.get("eps", 1e-8)
        beta1, beta2 = pg0.get("betas", (0.9, 0.999))
        wd = pg0.get("weight_decay", 0)
        # print(f"weight_decay {wd}")


        def rule(p):
            g = p.grad
            if g is None:
                return torch.zeros_like(p)

            st = self.optimizer.state.get(p, {})
            if fallback_to_neg_grad:
                return -p.grad
            elif (
                "exp_avg" in st
                and "exp_avg_sq" in st
                and st.get("step", 0) > 0
            ):

      
                m = st["exp_avg"]
                v = st["exp_avg_sq"]
                t = st["step"] + 1

                # mf = m.flatten()
                # vf = v.flatten()
                # gf = g.flatten()

                # print(
                #     f"t={t} | "
                #     f"g[:10]={gf[:10].tolist()} | "
                #     f"m[:10]={mf[:10].tolist()} | "
                #     f"v[:10]={vf[:10].tolist()} | "
                #     f"||g||={gf.norm().item():.3e} | "
                #     f"||m||={mf.norm().item():.3e} | "
                #     f"||v||={vf.norm().item():.3e}, t={t}"
                # )
                m_new = beta1 * m + (1 - beta1) * g
                v_new = beta2 * v + (1 - beta2) * (g * g)

                m_hat = m_new / (1 - beta1 ** t)
                v_hat = v_new / (1 - beta2 ** t)


                return -m_hat / (v_hat.sqrt() + eps) 
            else:
                gf = g.flatten()
                pf = p.flatten()

                term1 = g / (g.abs() + eps)      # g-normalized
                term2 = wd * p                  # weight decay term
                df = (term1 - term2).flatten()

                # print(
                #         f"(g/(|g|+eps) - wd*p)[:10]={df[:10].tolist()} | "
                #         f"g_norm[:10]={gf[:10].tolist()} | "
                #         f"wd*p[:10]={(wd*pf)[:10].tolist()} | "
                #         f"||g_norm||={term1.flatten().norm().item():.3e} | "
                #         f"||wd*p||={(wd*pf).norm().item():.3e}"
                #     )
                return - p.grad / (p.grad.abs() + eps) 
        return rule
    


    @torch.no_grad()
    def update_model(self, alpha):
        """
        Trial update for Muon-controlled params only.
        """
        for group_idx in self._muon_group_indices():
            group = self.optimizer.param_groups[group_idx]
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.mul_(1 - alpha * wd)
                p.add_(self.rule(p), alpha=alpha)

    @torch.no_grad()
    def restore_model(self, alpha):
        for group_idx in self._muon_group_indices():
            group = self.optimizer.param_groups[group_idx]
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    print("GRADIENT IS NONE!!! at 0")
                    continue
                p.add_(self.rule(p), alpha=-alpha)
                p.div_(1 - alpha * wd)

    @torch.no_grad()
    def update_adam_model(self):
        for group_idx in self._adam_group_indices():
            group = self.optimizer.param_groups[group_idx]
            lr = group.get("lr", self.start_lr)
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.mul_(1 - lr * wd)
                p.add_(self.rule(p), alpha=lr)

    @torch.no_grad()
    def restore_adam_model(self):
        for group_idx in self._adam_group_indices():
            group = self.optimizer.param_groups[group_idx]
            lr = group.get("lr", self.start_lr)
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(self.rule(p), alpha=-lr)
                p.div_(1 - lr * wd)
  


    # def perturb_parameters_global_(self, rho):
 
    #     paras = list(self.paras)

    #     noises = {}
    #     with torch.no_grad():
    #         device = "cuda"

    #         # 1) sample noise and compute its norm
    #         noise_norm_sq = torch.zeros((), device=device)
    #         for p in paras:
    #             if p.requires_grad:
    #                 z = torch.randn_like(p)
    #                 noises[p] = z
    #                 noise_norm_sq += z.pow(2).sum()

    #         noise_norm = noise_norm_sq.sqrt().add_(1e-12)
    #         scale = rho / noise_norm

    #         # 2) apply scaled noise
    #         for p in paras:
    #             if p.requires_grad:
    #                 z = noises[p]
    #                 z.mul_(scale)
    #                 p.add_(z)


    #     return noises


    # def restore_parameters_(self, noises):
    #     with torch.no_grad():
    #         for p, z in noises.items():
    #             p.sub_(z)


    def check_optimizer_step_vs_rule(
        self,
        optimizer,
        rule_fn,
        prefix="[LineSearchScheduler]"
    ):
        params = []
        param_groups = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
                    param_groups.append(group)

        if len(params) == 0:
            print(f"{prefix} no parameters found")
            return

        # save params
        old_params = [p.detach().clone() for p in params]
        old_grads = [
            None if p.grad is None else p.grad.detach().clone()
            for p in params
        ]
        old_states = []
        for p in params:
            state = optimizer.state.get(p, {})
            state_copy = {}
            for k, v in state.items():
                if torch.is_tensor(v):
                    state_copy[k] = v.detach().clone()
                else:
                    state_copy[k] = v
            old_states.append(state_copy)

        # ----- rule update -----
        with torch.no_grad():
            self.update_adam_model()
            for p, group in zip(params, param_groups):
                if p.grad is None:
                    continue
                if self._is_muon_group(group):
                    lr = group.get("lr", self.start_lr)
                    wd = group.get("weight_decay", 0.0)
                    p.mul_(1 - lr * wd)
                    p.add_(rule_fn(p), alpha=lr)

        rule_params = [p.detach().clone() for p in params]

        # restore
        for p, p_old in zip(params, old_params):
            p.data.copy_(p_old)
        for p, g_old in zip(params, old_grads):
            if g_old is None:
                p.grad = None
            else:
                if p.grad is None:
                    p.grad = g_old.clone()
                else:
                    p.grad.copy_(g_old)
        for p, state_old in zip(params, old_states):
            state = optimizer.state[p]
            state.clear()
            for k, v in state_old.items():
                state[k] = v.detach().clone() if torch.is_tensor(v) else v

        # ----- optimizer update -----
        optimizer.step()

        opt_params = [p.detach().clone() for p in params]

        # ----- compare -----
        max_diff = 0.0
        for r, o in zip(rule_params, opt_params):
            diff = (r - o).abs().max().item()
            max_diff = max(max_diff, diff)

        print(
            f"{prefix} max |param diff| = {max_diff:.3e}"
        )

        # restore original params
        for p, p_old in zip(params, old_params):
            p.data.copy_(p_old)
        for p, g_old in zip(params, old_grads):
            if g_old is None:
                p.grad = None
            else:
                if p.grad is None:
                    p.grad = g_old.clone()
                else:
                    p.grad.copy_(g_old)
        for p, state_old in zip(params, old_states):
            state = optimizer.state[p]
            state.clear()
            for k, v in state_old.items():
                state[k] = v.detach().clone() if torch.is_tensor(v) else v

    @torch.no_grad()
    def test_update_restore_max_diff(self, alpha):
        """
        Test max parameter difference after:
            update_model(alpha) -> closure(require_grad=False) -> restore_model(alpha)

        Returns:
            max |Δparam|
        """
        params = self._optimizer_params() if self.optimizer_type == "Muon" else self.paras
        backup = [p.detach().clone() for p in params]

        self.update_adam_model()
        self.update_model(alpha)
        self.restore_model(alpha)
        self.restore_adam_model()

        max_diff = 0.0
        for p, p0 in zip(params, backup):
            diff = (p.detach() - p0).abs().max().item()
            max_diff = max(max_diff, diff)

        print(f"[TEST] alpha={alpha:.3e}, max |param diff| = {max_diff:.3e}")
        return max_diff
    

    def clear_momentum(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state.get(p, {})
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                    

        

    def step(self, closure, condition="armijo", c1=0.6, factor=0.5, amax=1.0, amin=1e-6, step=0, interval=100, warmup_length=100, log_dir="./", start_lr=1):
        """
        condition: Line Search condition. Option: armijo,
        search_mode: Option: backtracking, forward, interpolate
        factor: used for searching. (growing/shrinking LR)
        c1: parameter for armijo rule
        c2: parameter for wolfe-condition
        interval: perform line search every {interval} steps.
        """
        k = step % interval 
        alpha = self.optimizer.param_groups[self.controlled_group_indices[0]]["lr"]
        if step < warmup_length:
            interval = warmup_length



        if k != 0 and step != warmup_length: 
            if self.prev_alpha >= self.line_search_alpha: 
                # progress in [0, 1]
                t = (k + 1) / interval
                # cosine interpolation (smooth start & end)
                cosine_frac = 0.5 * (1 - math.cos(math.pi * t))
                # lr = self.line_search_alpha
                lr = self.prev_alpha + cosine_frac * (
                    self.line_search_alpha - self.prev_alpha
                )
                for group_idx in self.controlled_group_indices:
                    self.optimizer.param_groups[group_idx]["lr"] = lr
                return
            warmup_frac = (k + 1) / interval
            lr = self.prev_alpha + warmup_frac * (self.line_search_alpha - self.prev_alpha)
            

            # print("denom", denom, "line_search_magnitude", self.line_search_magnitude, "prev_magnitude", self.prev_magnitude, "magnitude", self.magnitude, "lr", lr)
            for group_idx in self.controlled_group_indices:
                self.optimizer.param_groups[group_idx]["lr"] = lr
            return

        self.optimizer.zero_grad(set_to_none=True)

        loss = closure(require_grad=True)

        self.rule = self.get_potential_update_direction()

        muon_inner = 0.0
        adam_const = 0.0
        with torch.no_grad():
            for group_idx in self._muon_group_indices():
                group = self.optimizer.param_groups[group_idx]
                wd = group.get("weight_decay", 0.0)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    muon_inner += torch.sum(p.grad * self.rule(p))
                    muon_inner -= wd * torch.sum(p.grad * p)
            for group_idx in self._adam_group_indices():
                group = self.optimizer.param_groups[group_idx]
                lr_group = group.get("lr", self.start_lr)
                wd = group.get("weight_decay", 0.0)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    adam_const += lr_group * torch.sum(p.grad * self.rule(p))
                    adam_const -= lr_group * wd * torch.sum(p.grad * p)
        phi0 = loss
        derphi0 = muon_inner.detach()
        adam_const = adam_const.detach()

     
        # grad_sq = 0.0
        # dir_sq = 0.0
        # param_sq = 0.0
        # dot_gd = 0.0
        # dot_gp = 0.0

        # max_grad = 0.0
        # max_dir = 0.0

        # with torch.no_grad():
        #     for group_idx in self._muon_group_indices():
        #         group = self.optimizer.param_groups[group_idx]
        #         wd = group.get("weight_decay", 0.0)

        #         for p in group["params"]:
        #             if p.grad is None:
        #                 continue

        #             g = p.grad
        #             d = self.rule(p)


        #             # grad_sq += torch.sum(g * g).item()
        #             # dir_sq += torch.sum(d * d).item()
        #             # param_sq += torch.sum(p * p).item()

        #             # ===== dot product =====
        #             gd = torch.sum(g * d).item()
        #             gp = torch.sum(g * p).item()

        #             dot_gd += gd
        #             dot_gp += gp

        #             # ===== max element =====
        #             max_grad = max(max_grad, g.abs().max().item())
        #             max_dir = max(max_dir, d.abs().max().item())


        # grad_norm = grad_sq ** 0.5
        # dir_norm = dir_sq ** 0.5
        # param_norm = param_sq ** 0.5


        # cos_gd = dot_gd / (grad_norm * dir_norm + 1e-12)


        # inner = dot_gd - wd * dot_gp

        # phi0, derphi0 = loss, muon_inner


 
        # print("\n========== Line Search Debug ==========")
        # print(f"loss (phi0): {phi0}")
        # print(f"derphi0: {derphi0}")
        # print(f"adam_const: {adam_const}")

        # print("\n--- Norms ---")
        # print(f"grad_norm: {grad_norm:.6f}")
        # print(f"dir_norm:  {dir_norm:.6f}")
        # print(f"param_norm:{param_norm:.6f}")



        # self.test_update_restore_max_diff(alpha=alpha)
        # self.check_optimizer_step_vs_rule(
        #     optimizer=self.optimizer,
        #     rule_fn=self.rule,
        # )
        
        if derphi0 > 0: 
            # self.clear_momentum()
            print(f"ASCENT!!!, old derphi0 {derphi0}")
            self.rule = self.get_potential_update_direction(fallback_to_neg_grad=True)
            muon_inner = 0.0
            adam_const = 0.0
            with torch.no_grad():
                for group_idx in self._muon_group_indices():
                    group = self.optimizer.param_groups[group_idx]
                    wd = group.get("weight_decay", 0.0)
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        muon_inner += torch.sum(p.grad * self.rule(p))
                        muon_inner -= wd * torch.sum(p.grad * p)
                for group_idx in self._adam_group_indices():
                    group = self.optimizer.param_groups[group_idx]
                    lr_group = group.get("lr", self.start_lr)
                    wd = group.get("weight_decay", 0.0)
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        adam_const += lr_group * torch.sum(p.grad * self.rule(p))
                        adam_const -= lr_group * wd * torch.sum(p.grad * p)

            phi0, derphi0 = phi0, muon_inner.detach()
            adam_const = adam_const.detach()
            print(f"ASCENT!!!, new derphi0 {derphi0}")
            

        # xk = [p.detach().clone() for p in self.paras]
        # gk = [p.grad.detach().clone() if p.grad is not None else None for p in self.paras]
        @torch.no_grad()
        def phi(alpha):
            self.update_adam_model()
            self.update_model(alpha)
            val = closure(require_grad=False)
            self.restore_model(alpha)
            self.restore_adam_model()
            return val
    
        # def phi(alpha, n_samples=8, rel=0.02, alpha_min=1e-12, alpha_max=float("inf")):
        #     s = 0.0
        #     for _ in range(n_samples):
        #         z = torch.randn(()).item()  
        #         a = alpha * (1.0 + rel * z)
        #         a = max(alpha_min, min(a, alpha_max))
        #         s += single_phi(a)          
        #     return s / n_samples
        # ## This can be optimized 

        # print(f"start searching with alpha = {alpha0}, the prev_alpha is {self.prev_alpha}")
        alpha0 = self.line_search_alpha
        if step < warmup_length:
            alpha0 = 0.1
            num_search = self.num_search
        else:
            num_search = 1
    
        

        alpha, fc, _ = line_search_armijo(
                    f=phi,
                    derphi0=derphi0,
                    adam_const=adam_const,
                    phi0=phi0,
                    args=(),
                    c1=c1,
                    alpha0=alpha0,
                    num_search=num_search,
                    step=step,
                    search_mode=self.search_mode,
                    factor=factor,
                    log_dir=log_dir
                )
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"LINESEARCH LR: {alpha}")
        # if step < warmup_length and alpha < self.prev_alpha:
        #     alpha = self.prev_alpha
        
        # if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        #     current_lr = self.optimizer.param_groups[0]["lr"]
        #     alpha = float(current_lr if np.isfinite(current_lr) and current_lr > 0 else self.start_lr)

        # print(f"[LineSearchScheduler] alpha={alpha:.6g}, fc={fc}")
        
        # for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = alpha

        # if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        muon_indices = self._muon_group_indices()
        current_lr = self.optimizer.param_groups[muon_indices[0]]["lr"]
        # if step <= warmup_length and alpha < current_lr:
        #     alpha = current_lr

        # print(f"[LineSearchScheduler] alpha={alpha:.6g}, fc={fc}")
        
        # for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = alpha

        self.line_search_alpha = alpha

        # self.line_search_magnitude = alpha * dir_norm
        # if (not dist.is_initialized()) or dist.get_rank() == 0:
        #     print("LINESEARCH LR:", alpha, "magnitude:", self.line_search_magnitude)
        # if step < warmup_length:
        #     self.prev_magnitude = self.line_search_magnitude 
        # else:
        #     self.prev_magnitude = self.magnitude
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"LINESEARCH LR: {alpha}")
        # self.line_search_magnitude = alpha * dir_norm
        # if (not dist.is_initialized()) or dist.get_rank() == 0:
        #     print("LINESEARCH LR:", alpha, "magnitude:", self.line_search_magnitude)
        # if step < warmup_length:
        #     self.prev_magnitude = self.line_search_magnitude 
        # else:
        #     self.prev_magnitude = self.magnitude

        prev_lr = self.optimizer.param_groups[muon_indices[0]]["lr"]
        self.prev_alpha = prev_lr
        prev_lr = self.optimizer.param_groups[muon_indices[0]]["lr"]
        self.prev_alpha = prev_lr





def line_search_armijo(f, derphi0, adam_const, phi0, args=(), c1=1e-4, alpha0=1, num_search=16, step=0, search_mode="backtrack", factor=0.5, log_dir=""):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Parameters
    ----------
    f : callable
        Function to be minimized.
    xk : array_like
        Current point.
    pk : array_like
        Search direction.
    gfk : array_like
        Gradient of `f` at point `xk`.
    old_fval : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Optional arguments.
    c1 : float, optional
        Value to control stopping criterion.
    phi0 : scaler
        current loss
    derphi0 : scalar,
        inner product of dk and gradient
    alpha0 : scalar, optional
        Value of `alpha` at start of the optimization.

    Returns
    -------
    alpha
    f_count
    f_val_at_alpha

    Notes
    -----
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    """
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        value = f(alpha1)
        return value

    use_ddp = dist.is_initialized() 
    # print(f"USE DDP {use_ddp}")
    if use_ddp:
            alpha, phi1 = search_bisection_ddp(phi, phi0, derphi0, adam_const, c1=c1,
                                            old_alpha=alpha0, grow=1/factor, shrink=factor, amax=1, amin=1e-6, num_search=num_search)
            # alpha, phi1 = search_bisection_ddp_visual(phi, phi0, derphi0, c1=c1,
            #                                   old_alpha=alpha0, shrink=factor, grow=1/factor, amax=1, amin=1e-6, num_search=num_search, log_dir=log_dir, global_step=step)
    else:
            alpha, phi1 = search_bisection(phi, phi0, derphi0, adam_const, c1=c1,
                                            old_alpha=alpha0, grow=1/factor, shrink=factor, amax=1, amin=1e-6, num_search=num_search)
            # alpha, phi1 = search_bisection_visual(phi, phi0, derphi0, c1=c1,
            #                                               old_alpha=alpha0, shrink=factor, grow=1/factor, amax=1, amin=1e-6, num_search=num_search, log_dir=log_dir, global_step=step)
    
    

    return alpha, fc[0], phi1




def search_bisection_ddp_visual(phi, phi0, derphi0, c1,
                     old_alpha, grow=2.0, shrink=0.5,
                     amax=1, amin=1e-6, num_search=10,
                    t_min=0.0, t_max=3e-3, num_points=100, log_dir=None, global_step=0):
    
    use_ddp = dist.is_initialized() 
    ddp_on = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    flat_eps = 5e-2
    is_rank0 = (rank == 0)
    world_size = dist.get_world_size() if ddp_on else 1
    device = (
        torch.device("cuda")
        if use_ddp
        else None
    )
    alpha = old_alpha
    phi_a = phi(alpha)
    phi_old = phi_a
    explored = [] 
    import os
    loss_list = [phi0, phi_a]
    os.makedirs(log_dir, exist_ok=True)
    plot_path = os.path.join(log_dir, f"backtracking_ls_{global_step}.png")
    if rank == 0:
        armijo_old_work = phi_a <= phi0 + c1 * alpha * derphi0
    armijo_flag = torch.tensor(
                [int(armijo_old_work)] if rank == 0 else [0],
                device=device,
            )
    # print(f"[rank {rank}]  Before old armijo_broadcast")
    dist.broadcast(armijo_flag, src=0)
    # print(f"[rank {rank}]  After old armijo_broadcast")
    armijo_old_work = bool(armijo_flag.item())

    # # print(f'line search: old armijo={armijo_old},rank={rank}')
    if is_rank0:
        explored.append((alpha, phi_a))

    if armijo_old_work:
            for _ in range(num_search): 
            
                new_alpha = alpha * grow
                

                if rank == 0:
                    exceed = new_alpha >= amax
                exceed_flag = torch.tensor(
                    [int(exceed)] if rank == 0 else [0],
                    device=device,
                )
                # print(f"[rank {rank}]  Before  exceed_broadcast")
                dist.broadcast(exceed_flag, src=0)
                # print(f"[rank {rank}]  After  exceed_broadcast")
                exceed = bool(exceed_flag.item())
                
                # print(f'line search: exceed={exceed},rank={rank}')
                if exceed:
                    break

                new_phi = phi(new_alpha)
                if is_rank0:
                    explored.append((new_alpha, new_phi))
                # print(f'line search: loss={new_phi},rank={rank}')
                # print(f'line search: new alpha={new_alpha},rank={rank}')


                if rank == 0:
                    accept = new_phi > phi0 + c1 * new_alpha * derphi0
                accept_flag = torch.tensor(
                            [int(accept)] if rank == 0 else [0],
                            device=device,
                        )
                # print(f"[rank {rank}]  Before  accept_broadcast")
                dist.broadcast(accept_flag, src=0)
                # print(f"[rank {rank}]  After  accept_broadcast")
                accept = bool(accept_flag.item())
                # print(f'line search: accept={accept},rank={rank}')
                if accept:
                    break

        
                alpha = new_alpha
                phi_a = new_phi


    else:
            for _ in range(num_search): 
    
                new_alpha = alpha * shrink

                if rank == 0:
                    exceed = new_alpha <= amin
                exceed_flag = torch.tensor(
                    [int(exceed)] if rank == 0 else [0],
                    device=device,
                )
                # print(f"[rank {rank}]  Before  exceed_broadcast")
                dist.broadcast(exceed_flag, src=0)
                # print(f"[rank {rank}]  After  exceed_broadcast")
                exceed = bool(exceed_flag.item())
                # print(f'line search: exceed={exceed},rank={rank}')
                if exceed:
                    break


                new_phi = phi(new_alpha)
                loss_list.append(new_phi)
                if is_rank0:
                    explored.append((new_alpha, new_phi))
                # print(f'line search: loss={new_phi},rank={rank}')
                # print(f'line search: new alpha={new_alpha},rank={rank}')


                if rank == 0:
                    accept = new_phi <= phi0 + c1 * new_alpha * derphi0
                    if len(loss_list) >= 3:
                        rng = max(loss_list) - min(loss_list)
                        r_range = rng / max(abs(phi_a), 1e-12)
                        flat_region = r_range <= flat_eps
                    else:
                        flat_region = False

                accept_flag = torch.tensor(
                            [int(accept)] if rank == 0 else [0],
                            device=device,
                        )
                flat_flag = torch.tensor(
                        [int(flat_region)] if rank == 0 else [0],
                        device=device
                    )
                # print(f"[rank {rank}]  Before  accept_broadcast")
                dist.broadcast(accept_flag, src=0)
                # print(f"[rank {rank}]  After  accept_broadcast")
                accept = bool(accept_flag.item())
                # print(f'line search: accept={accept},rank={rank}')
                dist.broadcast(flat_flag, src=0)
                flat_region = bool(flat_flag.item())
                # if flat_region:
                #         print(f" Flat Region: {flat_region}. loss list: {loss_list}, maxdiff : {max(loss_list) - min(loss_list)}")
                #         return old_alpha, phi_old 
                if accept:
                    alpha = new_alpha
                    phi_a = new_phi
                    break
                


                alpha = new_alpha
                phi_a = new_phi
    def reduce_mean_scalar(x):
        if not ddp_on:
            return float(x) if not torch.is_tensor(x) else float(x.detach().item())
        if not torch.is_tensor(x):
            x = torch.tensor(float(x), device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            x = x.detach()
            if x.dim() != 0:
                x = x.reshape(())
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= world_size
        return float(x.item())
    
    phi0_g = reduce_mean_scalar(phi0)
    derphi0_g = reduce_mean_scalar(derphi0)


    t_vals = np.linspace(1e-6, min(1, 3e-3), num_points)

    phi_vals = []
    for t in t_vals:
        v_local = phi(float(t))
        v = reduce_mean_scalar(v_local)
        phi_vals.append(v)

    phi_vals = np.array(phi_vals)

    # ============================================================
    # Plot (rank0 only)
    # ============================================================
    if is_rank0:
        # print(f"phi_vals {phi_vals}")

        # -------- Armijo line --------
        armijo_line = phi0_g + c1 * t_vals * derphi0_g

        # ========================================================
        # ⭐ 截断范围：只显示 1e-3 以内
        # ========================================================
        t_max = 3e-3

        mask = t_vals <= t_max
        t_plot = t_vals[mask]
        phi_plot = phi_vals[mask]
        armijo_plot = armijo_line[mask]

        # ========================================================
        # Plot
        # ========================================================
        plt.figure(figsize=(8, 6))

        plt.plot(t_plot, phi_plot, label="phi(t)", linewidth=2)
        plt.plot(t_plot, armijo_plot, "--", label="Armijo line", linewidth=2)

        # -------- explored points --------
        for i, (a, v) in enumerate(explored):
            if a <= t_max:
                plt.scatter(a, v, color="red", s=60)
                plt.annotate(
                    "init" if i == 0 else f"bt {i}",
                    (a, v),
                    textcoords="offset points",
                    xytext=(5, 5),
                )

        # -------- chosen alpha --------
        if alpha <= t_max:
            plt.scatter(
                alpha,
                phi_a,
                color="blue",
                s=120,
                marker="x",
                label="chosen alpha",
            )

        # -------- axis & style --------
        plt.xlabel("t (step size)")
        plt.ylabel("phi(t)")
        plt.title("Backtracking Line Search Visualization (DDP)")
        plt.grid(True)
        plt.legend()


        plt.xlim(0.0, t_max)

        plt.savefig(plot_path, dpi=200)
        plt.close()

    return alpha, phi_a


def search_bisection_visual(phi, phi0, derphi0, c1,
                     old_alpha, grow=2.0, shrink=0.5,
                     amax=1, amin=1e-6, num_search=10,
                    t_min=0.0, t_max=3e-3, num_points=100, log_dir=None, global_step=0):
    """Non-DDP visual version of bisection/backtracking search.

    Mirrors `search_bisection_ddp_visual` but without torch.distributed calls.
    Produces a plot under `log_dir` named `backtracking_ls_{global_step}.png`.
    Returns chosen (alpha, phi(alpha)).
    """
    print("search start")
    alpha = old_alpha
    phi_a = phi(alpha)
    phi_old = phi_a
    explored = []
    import os
    loss_list = [phi0, phi_a]
    if log_dir is None:
        log_dir = "."
    os.makedirs(log_dir, exist_ok=True)
    plot_path = os.path.join(log_dir, f"backtracking_ls_{global_step}.png")

    # Determine whether old alpha satisfies Armijo
    armijo_old_work = phi_a <= phi0 + c1 * alpha * derphi0
    print("find")
    if armijo_old_work:
        print("work")
        explored.append((alpha, phi_a))
        for _ in range(num_search):
            print("maybe")
            new_alpha = alpha * grow
            if new_alpha >= amax:
                break
            new_phi = phi(new_alpha)
            explored.append((new_alpha, new_phi))
            if new_phi > phi0 + c1 * new_alpha * derphi0:
                break
            alpha = new_alpha
            phi_a = new_phi

    else:
        print("no work")
        explored.append((alpha, phi_a))
        for _ in range(num_search):
            print("maybe")
            new_alpha = alpha * shrink
            if new_alpha <= amin:
                break
            new_phi = phi(new_alpha)
            loss_list.append(new_phi)
            explored.append((new_alpha, new_phi))
            accept = new_phi <= phi0 + c1 * new_alpha * derphi0


            if accept:
                alpha = new_alpha
                phi_a = new_phi
                break


            alpha = new_alpha
            phi_a = new_phi

    # Build plot data
    print("start plot")
    t_vals = np.linspace(max(1e-12, t_min), min(1.0, t_max), num_points)
    phi_vals = np.array([phi(float(t)) for t in t_vals])

    # Plot (single-process)
    armijo_line = phi0 + c1 * t_vals * (derphi0 if not torch.is_tensor(derphi0) else float(derphi0))

    plt.figure(figsize=(8, 6))
    plt.plot(t_vals, phi_vals, label="phi(t)", linewidth=2)
    plt.plot(t_vals, armijo_line, "--", label="Armijo line", linewidth=2)

    for i, (a, v) in enumerate(explored):
        if a <= t_max:
            plt.scatter(a, v, color="red", s=60)
            plt.annotate("init" if i == 0 else f"bt {i}", (a, v), textcoords="offset points", xytext=(5, 5))

    if alpha <= t_max:
        plt.scatter(alpha, phi_a, color="blue", s=120, marker="x", label="chosen alpha")

    plt.xlabel("t (step size)")
    plt.ylabel("phi(t)")
    plt.title("Backtracking Line Search Visualization")
    plt.grid(True)
    plt.legend()
    plt.xlim(0.0, t_max)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("search end")

    return alpha, phi_a
                





def search_bisection_ddp(phi, phi0, derphi0, adam_const, c1,
                     old_alpha, grow=2.0, shrink=0.5,
                     amax=1, amin=1e-6, num_search=10,
                    plot_path="./img/backtracking_ls.png",
                    t_min=0.0, t_max=1e-4, num_points=100):

    use_ddp = dist.is_initialized() 
    rank = dist.get_rank() if use_ddp else 0
    flat_eps = 5e-2
    device = (
        torch.device("cuda")
        if use_ddp
        else None
    )
    alpha = old_alpha
    phi_a = phi(alpha)
    phi_old = phi_a

    loss_list = [phi0, phi_a]

    if rank == 0:
        armijo_old_work = phi_a <= phi0 + c1 * (alpha * derphi0 + adam_const)
    armijo_flag = torch.tensor(
                [int(armijo_old_work)] if rank == 0 else [0],
                device=device,
            )
    # print(f"[rank {rank}]  Before old armijo_broadcast")
    dist.broadcast(armijo_flag, src=0)
    # print(f"[rank {rank}]  After old armijo_broadcast")
    armijo_old_work = bool(armijo_flag.item())

    # # print(f'line search: old armijo={armijo_old},rank={rank}')

    if armijo_old_work:
            for _ in range(num_search): 
            
                new_alpha = alpha * grow

                if rank == 0:
                    exceed = new_alpha >= amax
                exceed_flag = torch.tensor(
                    [int(exceed)] if rank == 0 else [0],
                    device=device,
                )
                # print(f"[rank {rank}]  Before  exceed_broadcast")
                dist.broadcast(exceed_flag, src=0)
                # print(f"[rank {rank}]  After  exceed_broadcast")
                exceed = bool(exceed_flag.item())
                # print(f'line search: exceed={exceed},rank={rank}')
                if exceed:
                    break

                new_phi = phi(new_alpha)
                # print(f'line search: loss={new_phi},rank={rank}')
                # print(f'line search: new alpha={new_alpha},rank={rank}')


                if rank == 0:
                    accept = new_phi > phi0 + c1 * (new_alpha * derphi0 + adam_const)
                accept_flag = torch.tensor(
                            [int(accept)] if rank == 0 else [0],
                            device=device,
                        )
                # print(f"[rank {rank}]  Before  accept_broadcast")
                dist.broadcast(accept_flag, src=0)
                # print(f"[rank {rank}]  After  accept_broadcast")
                accept = bool(accept_flag.item())
                # print(f'line search: accept={accept},rank={rank}')
                if accept:
                    break

        
                alpha = new_alpha
                phi_a = new_phi

            return alpha, phi_a


    else:
            for _ in range(num_search): 
    
                new_alpha = alpha * shrink

                if rank == 0:
                    exceed = new_alpha <= amin
                exceed_flag = torch.tensor(
                    [int(exceed)] if rank == 0 else [0],
                    device=device,
                )
                # print(f"[rank {rank}]  Before  exceed_broadcast")
                dist.broadcast(exceed_flag, src=0)
                # print(f"[rank {rank}]  After  exceed_broadcast")
                exceed = bool(exceed_flag.item())
                # print(f'line search: exceed={exceed},rank={rank}')
                if exceed:
                    break


                new_phi = phi(new_alpha)
                loss_list.append(new_phi)
                # print(f'line search: loss={new_phi},rank={rank}')
                # print(f'line search: new alpha={new_alpha},rank={rank}')


                if rank == 0:
                    accept = new_phi <= phi0 + c1 * (new_alpha * derphi0 + adam_const)
                    if len(loss_list) >= 3:
                        rng = max(loss_list) - min(loss_list)
                        r_range = rng / max(abs(phi_a), 1e-12)
                        flat_region = r_range <= flat_eps
                    else:
                        flat_region = False

                accept_flag = torch.tensor(
                            [int(accept)] if rank == 0 else [0],
                            device=device,
                        )
                flat_flag = torch.tensor(
                        [int(flat_region)] if rank == 0 else [0],
                        device=device,
                    )
                # print(f"[rank {rank}]  Before  accept_broadcast")
                dist.broadcast(accept_flag, src=0)
                # print(f"[rank {rank}]  After  accept_broadcast")
                accept = bool(accept_flag.item())
                # print(f'line search: accept={accept},rank={rank}')
                dist.broadcast(flat_flag, src=0)
                flat_region = bool(flat_flag.item())
                # if flat_region:
                #         print(f" Flat Region: {flat_region}. loss list: {loss_list}, maxdiff : {max(loss_list) - min(loss_list)}")
                #         return old_alpha, phi_old 
                if accept:
                    return new_alpha, new_phi
                


                alpha = new_alpha
                phi_a = new_phi

            return alpha, phi_a


def search_bisection(phi, phi0, derphi0, adam_const, c1,
                     old_alpha, grow=2.0, shrink=0.5,
                     amax=1, amin=1e-6, num_search=10):

    alpha = old_alpha
    phi_a = phi(alpha)
    # phi_old = phi_a
    # flat_eps = 5e-2

    armijo_old = phi_a <= phi0 + c1 * (alpha * derphi0 + adam_const)
    # loss_list = [phi0, phi_a]
    # print(f"{armijo_old}, {phi_a}, {phi0}, {alpha}, {derphi0}")
    if armijo_old:
        for _ in range(num_search): 
        
            new_alpha = alpha * grow
            if new_alpha >= amax:
                break

            new_phi = phi(new_alpha)

            if new_phi > phi0 + c1 * (new_alpha * derphi0 + adam_const):
                break

    
            alpha = new_alpha
            phi_a = new_phi

        return alpha, phi_a


    else:
        for _ in range(num_search): 
            
   
            new_alpha = alpha * shrink
            if new_alpha <= amin:
                break
            new_phi = phi(new_alpha)
            # loss_list.append(new_phi)

        
            if new_phi <= phi0 + c1 * (new_alpha * derphi0 + adam_const):
                break



            alpha = new_alpha
            phi_a = new_phi

        return alpha, phi_a


def search_backtracking(phi, phi0, derphi0, c1, alpha, shrink, num_search):
    phi_a = phi(alpha)
    count = 0
    while phi_a > phi0 + c1 * alpha * derphi0 and count < num_search:
        alpha *= shrink
        phi_a = phi(alpha)
        count += 1
    return alpha, phi_a




def search_backtracking_visual(
    phi, phi0, derphi0,
    c1, alpha, shrink,
    plot_path="./img/backtracking_ls.png",
    t_min=0.0, t_max=1e-4, num_points=100,
    log_dir=None
):
    explored = []


    old_alpha = alpha
    phi_a = phi(alpha)
    phi_a_old = phi_a
    explored.append((alpha, phi_a))


    flat_eps = 2e-2     
    trend_eps = 1e-2    
    min_points = 4      
    delta = 1e-12

    loss_list = [phi0, phi_a]

    # ====== Backtracking loop ======
    while phi_a > phi0 + c1 * alpha * derphi0:

        alpha *= shrink
        phi_a = phi(alpha)
        explored.append((alpha, phi_a))
        loss_list.append(phi_a)

        # -------- 平台 / local-region 检测 --------
        if len(loss_list) >= min_points:
            rng = max(loss_list) - min(loss_list)
            r_range = rng / max(abs(phi0), delta)

            r_trend = abs(loss_list[-1] - loss_list[0]) / max(abs(phi0), delta)

            if (r_range <= flat_eps) and (r_trend <= trend_eps):
                print(
                    f"[flat-region detected] "
                    f"r_range={r_range:.3e}, r_trend={r_trend:.3e}, "
                    f"return old alpha={old_alpha}"
                )
                chosen_alpha, chosen_phi = old_alpha, phi_a_old
                break
    else:
        chosen_alpha, chosen_phi = alpha, phi_a

    
    t_vals = np.linspace(t_min, t_max, num_points)
    phi_vals_list = []
    for t in t_vals:
        value = phi(t)
        phi_vals_list.append(value)
    phi_vals = np.array(phi_vals_list)

    armijo_line = phi0 + c1 * t_vals * derphi0.item()

    plt.figure(figsize=(8, 6))
    plt.plot(t_vals, phi_vals, label="phi(t)", linewidth=2)
    plt.plot(t_vals, armijo_line, "--", label="Armijo line", linewidth=2)

    for i, (a, v) in enumerate(explored):
        plt.scatter(a, v, color="red", s=60)
        if i == 0:
            plt.annotate("init", (a, v), textcoords="offset points", xytext=(5, 5))
        else:
            plt.annotate(f"bt {i}", (a, v), textcoords="offset points", xytext=(5, 5))

    plt.scatter(
        chosen_alpha,
        chosen_phi,
        color="blue",
        s=120,
        marker="x",
        label="chosen alpha",
    )

    plt.xlabel("t (step size)")
    plt.ylabel("phi(t)")
    plt.title("Backtracking Line Search with Flat-Region Stop")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return chosen_alpha, chosen_phi





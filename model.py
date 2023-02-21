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

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

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
        elif isinstance(module, (LayerNorm, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
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
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

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
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class RLHF(nn.Module):
    def __init__(self, model, mode, prob_reward=False):
        super().__init__()
        self.model = model
        self.config = model.config

        # reward model
        self.n_embd = model.lm_head.in_features
        self.block_size = model.config.block_size
        # model.reward_head = nn.Sequential(nn.Linear(self.n_embd*self.block_size, self.n_embd*self.block_size), nn.ReLU(), nn.Linear(self.n_embd*self.block_size, 1, bias=False))
        model.policy_head = nn.Linear(model.lm_head.in_features, model.lm_head.out_features, bias=False)
        # model.policy_head = 
        self.mode = mode
        self.prob_reward = prob_reward
        if prob_reward:
            model.reward_head = nn.Linear(self.n_embd*self.block_size, 2, bias=False)
        else:
            model.reward_head = nn.Linear(self.n_embd*self.block_size, 1, bias=False)
    
    def forward_reward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.model.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.model.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        x = x.view(b, self.block_size*self.n_embd)

        rewards = self.model.reward_head(x)

        if self.prob_reward:
            probs = torch.softmax(rewards,1)
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                loss = F.cross_entropy(probs, targets, ignore_index=-1)
            else:
                loss = None
            return probs, loss
        else:
            return rewards
    
    def forward(self, idx, targets=None):
        if self.mode == 'reward':
            return self.forward_reward(idx, targets)
        elif self.mode == 'policy':
            return self.forward_policy(idx, targets)
        else:
            return self.model(idx, targets)
    
    def forward_policy(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.policy_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.policy_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    # @torch.no_grad()
    def generate(self, idx, max_new_tokens, device, block_size, use_reference=True, reward_model=None, hard_code_reward=True):
        # idx is (B, T) array of indices in the current context
        log_probs = torch.tensor([]).to(device)
        log_probs_ref = torch.tensor([]).to(device)
        values = torch.tensor([]).to(device)

        idx_cond_all = torch.zeros((idx.shape[0], block_size, max_new_tokens)).to(device)
        values_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)
        actions_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)
        rewards_all = torch.zeros((idx.shape[0],)).to(device)
        log_probs_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)
        advantages_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)
        returns_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)
        gamma = 1
        lam = 1

        # TODO: Critic, PPO
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # block_size = 256
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            
            probs_next = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs_next, num_samples=1) # (B, 1)

            probs_idx_next = torch.gather(probs_next, 1, idx_next)
            log_probs_idx_next = torch.log(probs_idx_next)
            log_probs = torch.cat((log_probs, log_probs_idx_next), dim=1)
            
            if use_reference:
                logits_ref, _ = self.model(idx_cond)
                logits_ref = logits_ref[:, -1, :] # becomes (B, C)
                probs_ref_next = F.softmax(logits_ref, dim=-1) # (B, C)
                probs_ref_idx_next = torch.gather(probs_ref_next, 1, idx_next)
                log_probs_ref_idx_next = torch.log(probs_ref_idx_next)
                log_probs_ref = torch.cat((log_probs_ref, log_probs_ref_idx_next), dim=1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            

            if i == max_new_tokens-1:
                states = idx[:,-max_new_tokens:]
                if hard_code_reward: 
                    # simple test where reward for outputting the letter 'z' (89)
                    rewards = torch.zeros_like(states, dtype=torch.float16)
                    rewards[states==89] = 1.0
                    rewards = torch.sum(rewards, 1, keepdim=True)
                    rewards[rewards > 1] = 1

                else:
                    if self.prob_reward:
                        rewards = reward_model.forward_reward(torch.tensor(states))[0][:,1].unsqueeze(-1)
                    else:
                        rewards = reward_model.forward_reward(torch.tensor(states))
                    

                for t in reversed(range(max_new_tokens)):
                    if t == max_new_tokens - 1:
                        # value at last state is 0
                        delta = rewards[:].squeeze() - values_all[:, t]
                        advantages_all[:, t] = delta
                        # returns_all[:, t] = rewards[:]
                    else:
                        # rewards can only be non-zero at the last state
                        delta = gamma * values_all[:, t + 1] - values_all[:, t]
                        advantages_all[:, t] = delta + gamma * lam * advantages_all[:, t + 1]
                        # returns_all[:, t] += gamma * returns_all[:, t + 1]

                    
            
        return idx[:,-max_new_tokens:], log_probs[:,-max_new_tokens:], log_probs_ref[:,-max_new_tokens:], rewards, advantages_all
    
    def generate_gumbel(self, idx, max_new_tokens, device, block_size, reward_model, use_reference=True):
        
        onehot = torch.tensor([]).to(device)
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # block_size = 256
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            
            probs_next = F.softmax(logits, dim=-1) # (B, C)
            #gumbel sample
            idx_next, onehot_next = self.gumbel_softmax(logits, tau=1, device=idx.device)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

            onehot = torch.cat((onehot, onehot_next), dim=2) # (B, T+1)

            if i == max_new_tokens-1:
                if self.prob_reward:
                    rewards = reward_model.forward_reward_gumbel(onehot)[0][:,1].unsqueeze(-1)
                else:
                    rewards = reward_model.forward_reward_gumbel(onehot)

        return idx[:,-max_new_tokens:], rewards

   
    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.distributions.Uniform(0,1).sample(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    def gumbel_softmax_sample(self, logits, tau, device, dim=1):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.shape).to(device)
        return F.softmax(y / tau, dim=dim)

    def gumbel_softmax(self, logits, tau, device):
        gumbel_sample = self.gumbel_softmax_sample(logits, tau, device)
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        onehot_idx_next = torch.nn.functional.one_hot(idx_next, num_classes=logits.shape[1]).squeeze()
        y = (onehot_idx_next-gumbel_sample).detach() + gumbel_sample
        idx_next_from_y = torch.argmax(y, dim=1).unsqueeze(-1)
        return idx_next_from_y, y.unsqueeze(-1)

    def forward_reward_gumbel(self, onehots, idx=None, targets=None):
        # (embd, vocab) @ (vocab, embd) = (embd,embd)
        
        device = onehots.device
        t = onehots.shape[2]
        b = onehots.shape[0]
        # b, t = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        # tok_emb = self.model.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        tok_emb = (self.model.transformer.wte.weight.T @ onehots)
        tok_emb = torch.transpose(tok_emb, 1, 2)

        if idx is not None:
            tok_emb2 = self.model.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            assert torch.all(tok_emb == tok_emb2)
        pos_emb = self.model.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.model.transformer.drop(tok_emb + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        x = x.view(b, self.block_size*self.n_embd)

        rewards = self.model.reward_head(x)

        if self.prob_reward:
            probs = torch.softmax(rewards,1)
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                loss = F.cross_entropy(rewards, targets, ignore_index=-1)
            else:
                loss = None
            return probs, loss
        else:
            return rewards
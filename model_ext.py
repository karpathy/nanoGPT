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

from utils import get_interval_values


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
        self.config = config  # <-- Store config on self
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        if self.config.scale_attn_by_context:
            # TODO - Per request one param per head
            self.log_attn_lambda = nn.Parameter(torch.zeros(self.n_head))

        # Flash attention is only available in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x, attention_mask = None):
        B, T, C = x.size()  # batch size, sequence length, embedding dim

        # 1) Compute q, k, v
        qkv = self.c_attn(x)  # shape (B, T, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2) Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3) Compute attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply the causal mask
            if attention_mask is not None:
                # Combine them: 0 => attend, 1 => ignore, or a bool tensor
                # Suppose 'self.bias==0' means "ignore," so you can do a logical OR:
                att_mask = (self.bias[:, :, :T, :T] == 0) | (attention_mask == False)
                att = att.masked_fill(att_mask, float('-inf'))
            else:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            # Optionally scale by context length * log(pos+1)
            if self.config.scale_attn_by_context:
                for pos in range(T):
                    for head_idx in range(self.n_head):
                        scale_factor = 1.0 + self.log_attn_lambda[head_idx] * math.log(pos + 1)
                        att[:, head_idx, pos, :] *= scale_factor

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_head, T, T) x (B, n_head, T, head_size)

        # 4) Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5) Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # ReLU vs. GELU
        if config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
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

    def forward(self, x, attention_mask = None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    activation: str = 'gelu'
    scale_attn_by_context: bool = False
    use_lstm_pos_enc: bool = False
    use_rotary_emb: bool = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer modules
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Optionally define LSTM for "positional encoding"
        if self.config.use_lstm_pos_enc:
            self.lstm_for_positions = nn.LSTM(
                input_size=config.n_embd,
                hidden_size=config.n_embd // 2,
                batch_first=True,
                bidirectional=True
            )

        # Language Model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)

        # Special scaled init for the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

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

    def forward(self, idx, targets=None, intervals=None, attention_mask = None):
        """
        idx:     (b, t) input token IDs
        targets: if shape (b, t), compute autoregressive training loss across the sequence
                 if shape (b, ), compute classification loss on the first position only
        intervals: optional, used if you want special interval-based indexing of logits
        """
        assert (intervals is None) or (targets is not None), "if you use intervals, you must provide targets"
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)

        if self.config.use_lstm_pos_enc:
            # Use the LSTM output instead of standard position embeddings
            lstm_out, _ = self.lstm_for_positions(tok_emb)
            x = self.transformer.drop(lstm_out)  # (b, t, n_embd)
        else:
            # Standard GPT position embeddings
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)  # (b, t, n_embd)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        x = self.transformer.ln_f(x)

        # Final logits
        logits = self.lm_head(x)  # (b, t, vocab_size)

        # Optional specialized interval-based loss
        if intervals is not None:
            target_logits = get_interval_values(logits, intervals - 1)  # -1 for "previous token"
            loss = F.cross_entropy(target_logits, targets)
        elif targets is not None:
            # If we have label tokens, compute standard cross-entropy
            if targets.dim() == 2:  # shape (b, t)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
            elif targets.dim() == 1:  # shape (b,)
                loss = F.cross_entropy(logits[:, 0, :], targets)
        else:
            # Inference-time optimization: only compute the last token's logits
            logits = logits[:, [-1], :]
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Truncate model's block_size (and embedding parameters) to the desired smaller size."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load GPT weights from a huggingface Transformers model (gpt2, gpt2-medium, etc.)
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt:", model_type)
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]  # discard buffer

        # Load HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not (k.endswith('.attn.masked_bias') or k.endswith('.attn.bias'))]

        # Transpose certain weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # gather all parameters that need grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed param tensors: {len(decay_params)}, total {num_decay_params:,} params")
        print(f"num non-decayed param tensors: {len(nodecay_params)}, total {num_nodecay_params:,} params")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = (fused_available and device_type == 'cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        flops_achieved = flops_per_iter * (1.0 / dt) if dt > 0 else 0
        flops_promised = 312e12  # A100 bfloat16 peak FLOPS is ~312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and
        complete the sequence max_new_tokens times.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)

            if temperature == 0:
                # Greedy decoding
                logits = logits[:, -1, :]
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature-based sampling
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if stop is not None and idx_next.item() in stop:
                break

        return idx

    @torch.no_grad()
    def classify(self, idx):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and return the argmax of the first token's logits.
        """
        logits, _ = self(idx)
        logits = logits[:, 0, :]
        return logits.argmax(dim=-1)

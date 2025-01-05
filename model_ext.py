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
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rotary_emb = config.use_rotary_emb
        self.rotary_emb = getattr(config, 'rotary_emb', None)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention.")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, position_ids=None, attn_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_rotary_emb:
            if position_ids is None:
                position_ids = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, T)
            cos, sin = self.rotary_emb(q, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal_mask = self.bias[:,:,:T,:T]
            if attn_mask is not None:
                # Combine causal_mask and attn_mask
                att = att.masked_fill((causal_mask == 0) | torch.isinf(attn_mask), float('-inf'))
            else:
                att = att.masked_fill(causal_mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # We do an 8× expansion, then split into two 4× halves (SwiGLU),
        # then project from 4× back to 1×.
        self.c_fc = nn.Linear(config.n_embd, 8 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        # Step 1: expand from n_embd -> 8*n_embd
        x_fc = self.c_fc(x)  # shape: (batch, seq_len, 8*n_embd)

        # Step 2: split into two 4× parts (a, b)
        a, b = x_fc.split(x_fc.size(-1) // 2, dim=-1)  # each (4*n_embd)

        # Step 3: SwiGLU
        x = nn.SiLU(b)  # shape: (batch, seq_len, 4*n_embd)

        # Step 4: project back to n_embd
        x = self.c_proj(x)     # shape: (batch, seq_len, n_embd)
        x = self.dropout(x)

        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Built-in RMSNorm in PyTorch >= 2.0
        # Note: By default eps=1e-5, but LLaMA typically uses eps=1e-6
        # and bias=False (elementwise_affine=True means there's a weight vector, but no bias term).
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=1e-6, elementwise_affine=True)

        # Your causal self-attention module (which may or may not match LLaMA exactly)
        self.attn = CausalSelfAttention(config)  

        # Another RMSNorm after attention
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=1e-6, elementwise_affine=True)

        # Your MLP module (SwiGLU or LLaMA’s triple-linear version)
        self.mlp = MLP(config)

    def forward(self, x, position_ids=None, attn_mask=None):
        h = self.input_layernorm(x)
        x = x + self.attn(h, position_ids=position_ids, attn_mask=attn_mask)

        h2 = self.post_attention_layernorm(x)
        x = x + self.mlp(h2)
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
    use_rotary_emb: bool = False  # NEW: Use rotary embeddings in attention layers

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # If not using rotary embeddings, keep wpe
            wpe = nn.Embedding(config.block_size, config.n_embd) if not config.use_rotary_emb else None,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # NEW: Initialize rotary embedding if requested
        if config.use_rotary_emb:
            # Here we assume LlamaRotaryEmbedding is defined and works similarly to HF's LLaMA
            # dimension per head:
            head_dim = config.n_embd // config.n_head
            self.rotary_emb = LlamaRotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.block_size,
                base=10000,
                scaling_factor=1.0,
                rope_type="default",
                config=None  # or pass a mock config if needed
            )
            # Assign to each attention layer
            for block in self.transformer.h:
                block.attn.rotary_emb = self.rotary_emb

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # For rotary embeddings, we don't add position embeddings
        if not self.config.use_rotary_emb:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos) 
            x = self.transformer.drop(tok_emb + pos_emb)
            position_ids = None
        else:
            tok_emb = self.transformer.wte(idx)
            x = self.transformer.drop(tok_emb)
            position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0).expand(b, t)

        # Convert attention_mask (if provided) into a suitable additive mask for attention
        # Attention mask should be of shape [b, t], where 1 means "attend" and 0 means "no attend".
        if attention_mask is not None:
            # Use a large negative number instead of -inf
            mask_value = -1e9
            attn_mask = (1.0 - attention_mask.view(b, 1, 1, t)) * mask_value
        else:
            attn_mask = None

        for block in self.transformer.h:
            x = block(x, position_ids=position_ids, attn_mask=attn_mask)  # pass the mask to each block

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # shift the logits and targets so that position i in logits predicts position i+1 in targets
            # Note: We slice off the last token in logits and the first token in targets so they align properly
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_targets = targets[:, 1:].contiguous()

            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_targets.view(-1),
                ignore_index=-100
            )
        else:
            # when targets is None, just return the logits for the last token
            # this remains the same
            logits = logits[:, [-1], :]
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, stop=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            if temperature == 0:
                # Greedy decoding: directly choose the token with the highest probability
                # no temperature scaling or sampling
                logits = logits[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature-based decoding
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)

            # append the chosen token index to the running sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)

            # if we sampled the stop token, return the sequence
            if stop is not None and idx_next.item() in stop:
                break

        return input_ids


    
    @torch.no_grad()
    def classify(self, idx):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and use the first position to predict a single token.
        """
        logits, _ = self(idx)
        logits = logits[:, 0, :]
        return logits.argmax(dim=-1)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def summary(self) -> str:
        """
        Returns a string summary of the core model configuration and
        parameter statistics, which you can compare to a Hugging Face LLaMA model.
        """
        # Extract relevant fields from self.config
        n_layer = getattr(self.config, 'n_layer', 'N/A')
        n_head = getattr(self.config, 'n_head', 'N/A')
        n_embd = getattr(self.config, 'n_embd', 'N/A')
        vocab_size = getattr(self.config, 'vocab_size', 'N/A')
        block_size = getattr(self.config, 'block_size', 'N/A')
        dropout = getattr(self.config, 'dropout', 'N/A')
        bias = getattr(self.config, 'bias', 'N/A')

        # Compute total parameters
        total_params = self.get_num_params(non_embedding=False)
        total_params_nonembed = self.get_num_params(non_embedding=True)

        # Build a summary string
        summary_str = (
            "======== Model Summary ========\n"
            f"Model class: {self.__class__.__name__}\n"
            f"Number of layers (n_layer): {n_layer}\n"
            f"Hidden size (n_embd): {n_embd}\n"
            f"Number of attention heads (n_head): {n_head}\n"
            f"Vocab size (vocab_size): {vocab_size}\n"
            f"Block size / context window (block_size): {block_size}\n"
            f"Dropout: {dropout}\n"
            f"Bias in linear/LN layers: {bias}\n"
            f"Total parameters (including embeddings): {total_params:,}\n"
            f"Total parameters (excluding positional embeddings): {total_params_nonembed:,}\n"
            "===============================\n"
        )
        return summary_str


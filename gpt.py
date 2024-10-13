from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.nn as nn
from pydantic.dataclasses import dataclass


def disable_torch_compile_if_amd(func):
    # Define a wrapper that applies the torch.compiler.disable decorator conditionally
    if torch.cuda.is_available() and "MI300X" in torch.cuda.get_device_name():
        return torch.compiler.disable()(func)
    else:
        return func


@disable_torch_compile_if_amd
def scaled_dot_product_attention_wrapper(q_BHTD, k_BHTD, v_BHTD, dropout_p=0.0, is_causal=True):
    # with torch.nn.attention.sdpa_kernel(
    #     enable_math=True,
    #     enable_flash=False,
    #     enable_mem_efficient=False
    # ):
    o_BHTD = F.scaled_dot_product_attention(q_BHTD, k_BHTD, v_BHTD, dropout_p=dropout_p, is_causal=is_causal)
    return o_BHTD

@dataclass
class GPTConfig:
    n_layers: int    # L
    n_heads: int     # H
    d_embd: int      # E
    max_seq_len: int = 1024
    vocab_size: int  = 50304 # V
    arch_name: str = 'gpt'

    @staticmethod
    def estimate_flops_per_token(model, config):
        # get param count
        N = sum(p.numel() for p in model.parameters())
                 
        head_dim = config['d_embd'] // config['n_heads'] 
         
        flops_per_token = 6 * N + 12 * config['n_layers'] * config['n_heads'] * head_dim * config['max_seq_len']
        
        return flops_per_token

    def __post_init__(self):
        assert self.d_embd % self.n_heads == 0, 'd_embd must be a multiple of n_heads.'


class CausalSelfAttention(nn.Module):
    def __init__(self, d_embd, n_heads, **kwargs):
        super().__init__()
        self.d_head = d_embd // n_heads  # D
        self.attn_proj = nn.Linear(d_embd, 3*d_embd)
        self.out_proj = nn.Linear(d_embd, d_embd)
 
    def forward(self, x_BTE):
        qkv = self.attn_proj(x_BTE).split(x_BTE.size(-1), -1)
        split_attn_head = lambda z: z.unflatten(-1, [-1, self.d_head]).transpose(1, 2)
        q_BHTD, k_BHTD, v_BHTD = map(split_attn_head, qkv)
        o_BHTD = scaled_dot_product_attention_wrapper(q_BHTD, k_BHTD, v_BHTD, dropout_p=0.0, is_causal=True)
        o_BTE = o_BHTD.transpose(1, 2).flatten(-2)
        y_BTE = self.out_proj(o_BTE)
        return y_BTE

    def load_ref_weights(self, ref):
        self.attn_proj.load_state_dict(ref.c_attn.state_dict())
        self.out_proj.load_state_dict(ref.c_proj.state_dict())


class GPTBlock(nn.Module):
    def __init__(self, d_embd, **kwargs):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_embd)
        self.attn = CausalSelfAttention(d_embd, **kwargs)
        self.ffn_norm = nn.LayerNorm(d_embd)
        self.ffn = nn.Sequential(
            nn.Linear(d_embd, 4*d_embd),
            nn.GELU(),
            nn.Linear(4*d_embd, d_embd)
        )

    def forward(self, x_BTE):
        x_BTE = x_BTE + self.attn(self.attn_norm(x_BTE))
        y_BTE = x_BTE + self.ffn(self.ffn_norm(x_BTE))
        return y_BTE

    def load_ref_weights(self, ref):
        self.attn_norm.load_state_dict(ref.ln_1.state_dict())
        self.attn.load_ref_weights(ref.attn)
        self.ffn_norm.load_state_dict(ref.ln_2.state_dict())
        self.ffn[0].load_state_dict(ref.mlp.c_fc.state_dict())
        self.ffn[2].load_state_dict(ref.mlp.c_proj.state_dict())


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, d_embd, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.pos_embd = nn.Embedding(max_seq_len, d_embd)
        self.tsfmr_blks = nn.ModuleList(GPTBlock(d_embd, **kwargs) for _ in range(n_layers))
        self.out_norm = nn.LayerNorm(d_embd)

    def forward(self, idx_BT):
        pos_T = torch.arange(idx_BT.size(1), dtype=torch.int64, device=idx_BT.device)
        x_BTE = self.tok_embd(idx_BT) + self.pos_embd(pos_T).unsqueeze(0)

        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE)

        x_BTE = self.out_norm(x_BTE)
        logits_BTV = x_BTE @ self.tok_embd.weight.T  # Weight tying

        return logits_BTV

    def load_ref_weights(self, ref):
        self.tok_embd.load_state_dict(ref.transformer.wte.state_dict())
        self.pos_embd.load_state_dict(ref.transformer.wpe.state_dict())

        for tsfmr_blk, ref_blk in zip(self.tsfmr_blks, ref.transformer.h):
            tsfmr_blk.load_ref_weights(ref_blk)

        self.out_norm.load_state_dict(ref.transformer.ln_f.state_dict())


if __name__ == '__main__':
    import json
    from pydantic import RootModel

    gpt2_125m = GPTConfig(
        arch_name='gpt',
        n_layers=12,
        n_heads=12,
        d_embd=768,
        max_seq_len=1024,
        vocab_size=50304,
    )
    with open('configs/gpt2-125m.json', 'w') as f:
        json.dump(RootModel[GPTConfig](gpt2_125m).model_dump(), f, indent=2)

    gpt2_350m = GPTConfig(
        arch_name='gpt',
        n_layers=24,
        n_heads=16,
        d_embd=1024,
        max_seq_len=1024,
        vocab_size=50304,
    )
    with open('configs/gpt2-350m.json', 'w') as f:
        json.dump(RootModel[GPTConfig](gpt2_125m).model_dump(), f, indent=2)

    gpt2_774m = GPTConfig(
        arch_name='gpt',
        n_layers=36,
        n_heads=20,
        d_embd=1280,
        max_seq_len=1024,
        vocab_size=50304,
    )
    with open('configs/gpt2-774m.json', 'w') as f:
        json.dump(RootModel[GPTConfig](gpt2_125m).model_dump(), f, indent=2)

    gpt2_xl = GPTConfig(
        arch_name='gpt',
        n_layers=48,
        n_heads=25,
        d_embd=1600,
        max_seq_len=1024,
        vocab_size=50304,
    )
    with open('configs/gpt2-1.5b.json', 'w') as f:
        json.dump(RootModel[GPTConfig](gpt2_xl).model_dump(), f, indent=2)

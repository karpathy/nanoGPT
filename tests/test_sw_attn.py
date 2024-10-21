from functools import lru_cache

import torch
import numpy as np
import torch.nn.functional as F
from mistral import create_sliding_window_attention_mask
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention
)


torch.set_default_device('cuda')
torch.manual_seed(3985)

B, H, T, D = 16, 16, 8192, 64
W = 1024

query = torch.rand([B, H, T, D], dtype=torch.bfloat16)
key = torch.rand_like(query)
value = torch.rand_like(query)
swa_mask = create_sliding_window_attention_mask(W, T)


flex_attention = torch.compile(flex_attention, dynamic=False)

def swa_mask_mod(b, h, q_idx, kv_idx):
    causal_mask = (q_idx >= kv_idx)
    window_mask = (q_idx - kv_idx < W)  # attention-gym says <= but I think it's wrong
    return causal_mask & window_mask

@lru_cache
def create_block_mask_cached(*args, **kwargs):
    blk_mask = create_block_mask(*args, **kwargs, device='cuda')
    return blk_mask

blk_mask = create_block_mask_cached(swa_mask_mod, 1, 1, T, T)


def test_swa_mask():
    swa_mask_fa = swa_mask_mod(1, 1, torch.arange(T).unsqueeze(1), torch.arange(T).unsqueeze(0))
    assert torch.equal(swa_mask, swa_mask_fa), f'{swa_mask=}\n\n{swa_mask_fa=}'


def test_density():
    density = (W * (W - 1) / 2 + (T - W + 1) * W) / (T * T)
    density_g = swa_mask.count_nonzero() / swa_mask.numel()
    # density_g = 1.0 - blk_mask.sparsity() / 100  # Block sparsity counts non-zero blocks instead of elements
    assert np.isclose(density, density_g.item()), f'{density=}, {density_g=}'


def test_correctness():
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=swa_mask)
    output_g = flex_attention(query, key, value, block_mask=blk_mask)
    torch.testing.assert_close(output, output_g)

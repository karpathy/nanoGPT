from dataclasses import dataclass
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F




@dataclass
class GPTCfg:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_head: int = 4
    n_embed: int = 512
    dropout: float = 0.1
    bais = False


class CausalSelfAttention(nn.Module):
   
    def __init__(self, config: GPTCfg) -> None:
        super().__init__()
        # Config Params
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # Transformations 
        self.c_att = nn.Linear(config.n_embed, 3 * config.n_embed, bias = config.bias)
        self.att_dropout = nn.Dropout(config.dropout)
        self.head_dropout = nn.Dropout(config.dropout)
        # Register Buffer for att mask 
        if not self.use_flash:
            self.register_buffer('bias', torch.tril(torch.ones(self.block_size, self.block_size)
                                                    .view(1, 1, self.block_size, self.block_size)))


    def foward(self, x):

        B, T, C = x.size()
        
        q, k, v = self.c_att(x).chunk(3, dim=-1)
        q = q.view(B, self.n_head, T, C // self.n_head)  # B, nh, T, hs
        k = k.view(B, self.n_head, T, C // self.n_head)  # B, nh, T, hs
        v = v.view(B, self.n_head, T, C // self.n_head)  # B, nh, T, hs

        # scaled dot product attention
        if self.use_flash:
            # Pytorch implementation of FlashAttention
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            scale_factor = 1.0 / math.sqrt(k.size(-1))
            att = q @ k.transpose(-2, -1) * scale_factor  # B, nh, T, T
            att = att.masked_fill(self.bias[:,:,:T,:T]==0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.att_dropout(att)
            y = att @ v  # B, nh, T, hs

        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.head_dropout(y)

        return y
        

            


        
        
        

    
    

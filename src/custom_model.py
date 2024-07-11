import sys
import os

sys.path.append('build/lib.linux-x86_64-3.10')
import h100_train as tk_train

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import Event
from torch.autograd import Function
from einops import rearrange


class AttentionFunction(Function):
    def forward(ctx, q, k, v, outputs, l_vec,  grad_q, grad_k, grad_v, d_vec):        
        assert q.shape[3] == 64, "TK train currently supports head dim 64 only"

        # Reset gradient tensors
        grad_q.zero_()
        grad_k.zero_()
        grad_v.zero_()

        tk_train.attention_train_forward_causal(q, k, v, outputs, l_vec)
        ctx.save_for_backward(q, k, v, outputs, l_vec, grad_q, grad_k, grad_v, d_vec)
        return outputs

    def backward(ctx, grad_output):        
        assert grad_output.shape[3] == 64, "TK train currently supports head dim 64 only"
        
        q, k, v, o, l_vec, grad_q, grad_k, grad_v, d_vec = ctx.saved_tensors

        # print("Inside backwards tk")
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()
        l_vec = l_vec.contiguous()
        grad_q = grad_q.contiguous()
        grad_k = grad_k.contiguous()
        grad_v = grad_v.contiguous()
        d_vec = d_vec.contiguous()
        
        tk_train.attention_train_backward_causal(
            q, k, v, o, 
            l_vec, d_vec, 
            grad_output.contiguous(), 
            grad_q, grad_k, grad_v
        )

        return grad_q, grad_k, grad_v, None, None, None, None, None, None


class CustomAttention(nn.Module):
    def __init__(self, config):
        super(CustomAttention, self).__init__()
        self.b = config.batch_size
        self.h = config.n_head
        self.n = config.block_size
        self.d = config.n_embd
        self.scale = 1 / (self.d ** 0.5)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        with torch.no_grad():
            self.outputs = torch.empty((self.b, self.h, self.n, self.d //self. h), dtype=torch.bfloat16, device='cuda', requires_grad=False)
            self.l_vec = torch.empty((self.b, self.h, self.n, 1), dtype=torch.bfloat16, device='cuda', requires_grad=False)
            self.grad_q = torch.empty((self.b, self.h, self.n, self.d // self.h), dtype=torch.bfloat16, device='cuda', requires_grad=False)
            self.grad_k = torch.empty((self.b, self.h, self.n, self.d // self.h), dtype=torch.bfloat16, device='cuda', requires_grad=False)
            self.grad_v = torch.empty((self.b, self.h, self.n, self.d // self.h), dtype=torch.bfloat16, device='cuda', requires_grad=False)
            self.d_vec = torch.empty((self.b, self.h, self.n, 1), dtype=torch.bfloat16, device='cuda', requires_grad=False)


    def forward(self, x):
        B, T, C = x.size() 
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d, dim=2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)
        

        output = AttentionFunction.apply(q, k, v, self.outputs, self.l_vec, self.grad_q, self.grad_k, self.grad_v, self.d_vec)
        y = output.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y



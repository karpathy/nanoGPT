
import torch
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k, sinusoidal_pos):
    seq_len, hidden_size = sinusoidal_pos.shape
    cos_pos = sinusoidal_pos[:, None, 1::2].repeat_interleave(2, dim=-1)
    sin_pos = sinusoidal_pos[:, None, ::2].repeat_interleave(2, dim=-1)

    # Decompose q & k into their respective sine and cosine parts
    q_cos = q * cos_pos
    q_sin = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
    q = q_cos + q_sin * sin_pos

    k_cos = k * cos_pos
    k_sin = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k)
    k = k_cos + k_sin * sin_pos

    # Compute attention
    a = torch.einsum('bthd,bshd->bhts', q, k)
    return a

# Example usage:
# batch_size, seq_len, num_heads, hidden_size = qw.shape
# sinusoidal_pos = get_sinusoidal_pos(seq_len, hidden_size)
# a = apply_rotary_pos_emb(qw, kw, sinusoidal_pos)

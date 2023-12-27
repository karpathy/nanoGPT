import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd

        # Register frequencies directly as buffers
        self.register_buffer('freq_left', (10000 ** (torch.arange(0, self.dim//2).float() / self.dim//2)))
        self.register_buffer('freq_right',(10000 ** (torch.arange(0, self.dim//2).float() / self.dim//2)))

    def forward(self, x):
        seq_len = x.shape[-2]
        device = x.device

        t = torch.arange(seq_len, device=device)

        # Get separate frequencies for left and right
        freqs_left = torch.einsum('i,j->ij', t, self.freq_left)
        freqs_right = torch.einsum('i,j->ij', t, self.freq_right)

        # Apply frequencies
        x_left, x_right = x[..., :self.dim//2], x[..., self.dim//2:]
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Combine the left and right parts back
        x = torch.cat([x_left, x_right], dim=-1)

        return x

class ShortRope(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n = config.shortrope_length
        self.dim = config.n_embd

        # Generate freqs of size n rather than full dim
        self.register_buffer('freq_left', (10000 ** (torch.arange(0, self.n//2).float() / self.n//2)))
        self.register_buffer('freq_right', (10000 ** (torch.arange(0, self.n//2).float() / self.n//2)))

    def forward(self, x):
        # Step 1: Get the input tensor shape
        batch_size, seq_len, _ = x.shape

        # Step 2: Split the input tensor into unrotated and rotated sections
        x_unrotated = x[..., :-self.n]  # All but the last n dimensions
        x_rotated = x[..., -self.n:]    # Only the last n dimensions

        # Step 3: Generate rotation frequencies
        t = torch.arange(self.n, device=x.device)
        freqs_left = torch.einsum('i,j->ij', t, self.freq_left)
        freqs_right = torch.einsum('i,j->ij', t, self.freq_right)

        # Calculate how many times to repeat freqs along the sequence length
        num_repeats = seq_len // self.n + int(seq_len % self.n != 0)

        # Repeat the frequency tensors to match the sequence length
        freqs_left = freqs_left.repeat(batch_size, num_repeats, 1)
        freqs_right = freqs_right.repeat(batch_size, num_repeats, 1)

        # Trim the excess elements so the freqs tensors match the sequence length
        freqs_left = freqs_left[:, :seq_len, :]
        freqs_right = freqs_right[:, :seq_len, :]

        # Step 4: Process the x_rotated section
        x_left = x_rotated[..., :self.n//2]
        x_right = x_rotated[..., self.n//2:]

        # Apply the cosine and sine rotations
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Invert the order of the right tensor's last dimension and negate it
        x_right = torch.flip(x_right, dims=[-1]) * -1

        # Combine the left and right rotated sections
        x_rotated = torch.cat([x_left, x_right], dim=-1)

        # Step 5: Combine the rotated and unrotated sections
        x = torch.cat([x_unrotated, x_rotated], dim=-1)

        return x


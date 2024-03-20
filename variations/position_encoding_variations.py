import torch
import torch.nn as nn
import math

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

class SymmetricalOverlapAngularPositions(nn.Module):
    """ Soap is a fresh and 'clean' implementation of Rotary Embeddings

    Symmetries and rotational overlap to optimize storage requiremnets.

    Using symmetries, reduces the cache size in specialized hardware
    by a factor of 8. (x <-> y symmetries, x-axis symmetries, y-axis symmetries)

    Applies same rotation to each pair of elements per vector.

    Likely to generalize with minimal finetuning via interpolation.
    """
    def __init__(self, config, size=None, num_angles=256):
        super().__init__()

        self.dim = size
        assert self.dim % 2 == 0, "Target length dim of must be even for rotary embeddings"

        self.num_angles = num_angles

        # Create a list of angles, from zero up to, not including 2*pi
        angles = torch.linspace(0, 2 * math.pi - ( 2 * math.pi/self.num_angles), steps=self.num_angles)
        self.register_buffer('angles', angles)

        self.first_pass = True

    def forward(self, x):
        seq_len = x.shape[-2]
        device = x.device

        # utilize 0 for first pass
        if not self.first_pass:
            self.angles = torch.roll(self.angles, shifts=1, dims=0)
        self.first_pass = False

        # Create index list, wrap around as necessary
        angle_indices = torch.arange(seq_len, device=device) % self.num_angles

        # Assign angles
        selected_angles = self.angles[angle_indices]

        # Run angles through sine and cosines
        sin_angles = selected_angles.sin().unsqueeze(-1).repeat(1, self.dim // 2)
        cos_angles = selected_angles.cos().unsqueeze(-1).repeat(1, self.dim // 2)

        # Split input tensor into even and odd components
        x_even, x_odd = x[..., ::2], x[..., 1::2]

        # Rotate components
        x_rotated_even = x_even * cos_angles - x_odd * sin_angles
        x_rotated_odd = x_even * sin_angles + x_odd * cos_angles

        # Reassemble rotated components
        x_combined = torch.empty_like(x, device=device)
        x_combined[... ,::2], x_combined[..., 1::2] = x_rotated_even, x_rotated_odd
        # TODO shorten x_combined to manually set length, then replace corresponding portion of original x matrix to implement shortrope

        return x_combined

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


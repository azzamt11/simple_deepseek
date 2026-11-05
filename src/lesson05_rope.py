"""Lesson 05: rotary positional embeddings (RoPE)."""

import torch


def rotate_half(x):
    """Rotate each consecutive pair [x0, x1] into [-x1, x0]."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def rotary_embedding(x, position_ids=None, base=10000.0):
    """Apply RoPE to x shaped [batch, sequence, hidden]."""
    batch_size, sequence_length, hidden_size = x.shape
    if hidden_size % 2 != 0:
        raise ValueError("hidden_size must be even for RoPE")

    if position_ids is None:
        position_ids = torch.arange(sequence_length, device=x.device)
    position_ids = position_ids.to(device=x.device)

    # One frequency for each pair of hidden features.
    pair_indices = torch.arange(0, hidden_size, 2, device=x.device)
    inverse_frequencies = base ** (-pair_indices.float() / hidden_size)
    angles = position_ids[:, None].float() * inverse_frequencies[None, :]

    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    # RoPE(x, p) = x cos(theta_p) + rotate_half(x) sin(theta_p).
    return x * cos + rotate_half(x) * sin


def main():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 4, 8)
    rotated = rotary_embedding(hidden_states)

    print("input shape:", hidden_states.shape)
    print("rotated shape:", rotated.shape)
    assert rotated.shape == hidden_states.shape

    # RoPE rotates feature pairs; it does not change their Euclidean norm.
    before_norm = torch.linalg.vector_norm(hidden_states, dim=-1)
    after_norm = torch.linalg.vector_norm(rotated, dim=-1)
    assert torch.allclose(before_norm, after_norm, atol=1e-6)
    print("pairwise norm preservation: passed")


if __name__ == "__main__":
    main()

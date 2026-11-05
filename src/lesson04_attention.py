"""Lesson 04: causal self-attention from first principles."""

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """A small, single-head causal self-attention layer."""

    def __init__(self, hidden_size):
        super().__init__()
        # Q = H W_Q + b_Q, K = H W_K + b_K, V = H W_V + b_V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, return_intermediates=False):
        hidden_size = hidden_states.shape[-1]

        # Projection equations: Q, K, V each have shape [B, S, D].
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # L = Q K^T: [B, S, D] @ [B, D, S] -> [B, S, S].
        scores = query @ key.transpose(-2, -1)
        # L_tilde = L / sqrt(D).
        scores = scores / math.sqrt(hidden_size)

        # M[i, j] is 0 for j <= i and -inf for future positions j > i.
        sequence_length = hidden_states.shape[-2]
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        # L_hat = L_tilde + M.
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # A = softmax(L_hat), row-wise over the keys.
        weights = torch.softmax(scores, dim=-1)
        # O = A V: [B, S, S] @ [B, S, D] -> [B, S, D].
        output = weights @ value

        if return_intermediates:
            return output, {
                "query": query,
                "key": key,
                "value": value,
                "scores": scores,
                "weights": weights,
                "causal_mask": causal_mask,
            }
        return output


def main():
    torch.manual_seed(0)
    batch_size, sequence_length, hidden_size = 2, 4, 8
    hidden_states = torch.randn(batch_size, sequence_length, hidden_size)

    attention = CausalSelfAttention(hidden_size)
    output, tensors = attention(hidden_states, return_intermediates=True)

    print("hidden_states shape:", hidden_states.shape)
    for name in ("query", "key", "value", "scores", "weights"):
        print(f"{name} shape:", tensors[name].shape)
    print("output shape:", output.shape)

    assert output.shape == hidden_states.shape
    assert tensors["query"].shape == (batch_size, sequence_length, hidden_size)
    assert tensors["scores"].shape == (batch_size, sequence_length, sequence_length)
    assert torch.allclose(
        tensors["weights"].sum(dim=-1), torch.ones(batch_size, sequence_length)
    )

    # The paper's causality test: changing future positions cannot change O(H)_0.
    first = torch.randn(1, 4, 4)
    second = first.clone()
    second[:, 1:, :] = torch.randn(1, 3, 4)
    test_attention = CausalSelfAttention(4)
    first_output = test_attention(first)
    second_output = test_attention(second)

    assert torch.allclose(first_output[:, 0, :], second_output[:, 0, :])
    assert not torch.allclose(first_output[:, 2, :], second_output[:, 2, :])
    print("causal invariance test: passed")


if __name__ == "__main__":
    main()

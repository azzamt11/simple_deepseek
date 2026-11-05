"""Practice: demonstrate that causal attention cannot see future positions."""

import torch

from lesson04_attention import CausalSelfAttention


def main():
    # Both inputs have the same complete hidden vector at position 0.
    first = torch.tensor(
        [[[1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0]]]
    )
    second = first.clone()

    # Change only future positions. Position 0 remains identical.
    second[:, 1:, :] = torch.tensor(
        [[[30.0, 40.0],
          [50.0, 60.0]]]
    )

    attention = CausalSelfAttention(hidden_size=2)
    first_output = attention(first)
    second_output = attention(second)

    # Expected input and output shapes are [batch, sequence, hidden].
    assert first.shape == (1, 3, 2)
    assert second.shape == (1, 3, 2)
    assert first_output.shape == first.shape
    assert second_output.shape == second.shape

    # Position 0 cannot see positions 1 or 2, so its output must be unchanged.
    position_zero_unchanged = torch.allclose(
        first_output[:, 0, :],
        second_output[:, 0, :],
    )
    assert position_zero_unchanged

    # Position 2 can see the changed positions, so it should change.
    position_two_changed = not torch.allclose(
        first_output[:, 2, :],
        second_output[:, 2, :],
    )
    assert position_two_changed

    print("first input:\n", first)
    print("second input:\n", second)
    print("position 0 unchanged:", position_zero_unchanged)
    print("position 2 changed:", position_two_changed)
    print("output shape:", first_output.shape)


if __name__ == "__main__":
    main()

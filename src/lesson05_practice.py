"""Practice: verify that RoPE depends on position."""

import torch

from lesson05_rope import rotary_embedding


def main():
    # The same vector receives different rotations at different positions.
    x = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    position_zero = torch.tensor([0])
    position_one = torch.tensor([1])

    rotated_zero = rotary_embedding(x, position_ids=position_zero)
    rotated_one = rotary_embedding(x, position_ids=position_one)

    assert x.shape == (1, 1, 4)
    assert rotated_zero.shape == x.shape
    assert rotated_one.shape == x.shape
    assert torch.allclose(rotated_zero, x)
    assert not torch.allclose(rotated_zero, rotated_one)

    print("position 0 output:\n", rotated_zero)
    print("position 1 output:\n", rotated_one)
    print("RoPE position test: passed")


if __name__ == "__main__":
    main()

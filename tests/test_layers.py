"""Tests for BitLinear layer."""

import mlx.core as mx
import mlx.nn as nn

from onebit.layers import BitLinear


def test_bitlinear_from_linear():
    """BitLinear should produce output with correct shape."""
    linear = nn.Linear(128, 64)
    mx.eval(linear.parameters())

    bitlinear = BitLinear.from_linear(linear, use_metal_kernel=False)

    x = mx.random.normal((1, 128))
    out = bitlinear(x)
    mx.eval(out)

    assert out.shape == (1, 64), f"Expected (1, 64), got {out.shape}"


def test_bitlinear_batch():
    """BitLinear should handle batched input."""
    linear = nn.Linear(64, 32)
    mx.eval(linear.parameters())

    bitlinear = BitLinear.from_linear(linear, use_metal_kernel=False)

    x = mx.random.normal((4, 10, 64))  # batch=4, seq=10, dim=64
    out = bitlinear(x)
    mx.eval(out)

    assert out.shape == (4, 10, 32), f"Expected (4, 10, 32), got {out.shape}"


def test_bitlinear_packed_shape():
    """Packed weights should be 4x smaller in last dim."""
    bitlinear = BitLinear(128, 64)
    assert bitlinear.packed_weights.shape == (64, 32)  # 128 / 4 = 32


def test_bitlinear_with_bias():
    """BitLinear with bias should work."""
    linear = nn.Linear(64, 32)
    # Manually add bias
    mx.eval(linear.parameters())

    bitlinear = BitLinear.from_linear(linear, use_metal_kernel=False)

    x = mx.random.normal((1, 64))
    out = bitlinear(x)
    mx.eval(out)
    assert out.shape == (1, 32)


if __name__ == "__main__":
    test_bitlinear_from_linear()
    test_bitlinear_batch()
    test_bitlinear_packed_shape()
    test_bitlinear_with_bias()
    print("All tests passed!")

"""Tests for ternary quantization."""

import mlx.core as mx
from onebit.quant import pack_ternary, unpack_ternary, quantize_to_ternary


def test_pack_unpack_roundtrip():
    """Pack and unpack should be lossless for ternary values."""
    M, K = 16, 64
    # Create random ternary weights
    weights = mx.random.randint(-1, 2, (M, K)).astype(mx.int8)
    packed = pack_ternary(weights)
    unpacked = unpack_ternary(packed, K)

    assert unpacked.shape == (M, K)
    expected = weights.astype(mx.float16)
    assert mx.allclose(unpacked, expected).item(), "Pack/unpack roundtrip failed"


def test_pack_shape():
    """Packed weights should be 4x smaller."""
    M, K = 32, 128
    weights = mx.random.randint(-1, 2, (M, K)).astype(mx.int8)
    packed = pack_ternary(weights)
    assert packed.shape == (M, K // 4)
    assert packed.dtype == mx.uint8


def test_quantize_to_ternary():
    """quantize_to_ternary should produce packed weights and a scale."""
    weight = mx.random.normal((64, 128)) * 0.1
    packed, scale = quantize_to_ternary(weight)

    assert packed.shape == (64, 128 // 4)
    assert packed.dtype == mx.uint8
    assert scale.shape == (1,)
    assert scale.dtype == mx.float16
    assert scale.item() > 0


def test_unpack_values():
    """Unpacked values should only be -1, 0, or +1."""
    weight = mx.random.normal((32, 64)) * 0.5
    packed, scale = quantize_to_ternary(weight)
    unpacked = unpack_ternary(packed, 64)

    unique_vals = set(unpacked.reshape(-1).tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0}), f"Unexpected values: {unique_vals}"


def test_specific_values():
    """Test with known ternary values."""
    weights = mx.array([[1, -1, 0, 1, -1, 0, 1, -1]], dtype=mx.int8)
    packed = pack_ternary(weights)
    unpacked = unpack_ternary(packed, 8)

    expected = mx.array([[1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0]], dtype=mx.float16)
    assert mx.allclose(unpacked, expected).item(), f"Got {unpacked.tolist()}"


if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_pack_shape()
    test_quantize_to_ternary()
    test_unpack_values()
    test_specific_values()
    print("All tests passed!")

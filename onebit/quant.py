"""Ternary quantization: pack/unpack weights to/from {-1, 0, +1}."""

import mlx.core as mx


def quantize_to_ternary(weight: mx.array, per_row: bool = True) -> tuple[mx.array, mx.array]:
    """Quantize FP16/BF16 weight matrix to ternary {-1, 0, +1}.

    Uses absmean quantization from the BitNet b1.58 paper.
    Per-row scaling preserves much more information than per-tensor.

    Args:
        weight: [out_features, in_features] float16/bfloat16/float32
        per_row: If True, use per-row scale (recommended). Otherwise per-tensor.

    Returns:
        packed: [out_features, in_features // 4] uint8 — 4 ternary values per byte
        scale:  [out_features] or [1] float16 — scale factor(s)
    """
    if per_row and weight.ndim == 2:
        # Per-row scaling: each output neuron gets its own scale
        scale = mx.mean(mx.abs(weight), axis=1, keepdims=True)  # [M, 1]
        scale = mx.maximum(scale, mx.array(1e-5))
        normalized = weight / scale
        ternary = mx.clip(mx.round(normalized), -1, 1).astype(mx.int8)
        packed = pack_ternary(ternary)
        return packed, scale.squeeze(1).astype(mx.float16)  # [M]
    else:
        # Per-tensor scaling (original BitNet paper, for natively-trained models)
        scale = mx.mean(mx.abs(weight))
        scale = mx.maximum(scale, mx.array(1e-5))
        normalized = weight / scale
        ternary = mx.clip(mx.round(normalized), -1, 1).astype(mx.int8)
        packed = pack_ternary(ternary)
        return packed, scale.astype(mx.float16).reshape(1)


def pack_ternary(weights: mx.array) -> mx.array:
    """Pack ternary weights {-1, 0, +1} into 2-bit packed uint8.

    Encoding per 2 bits: 00 = 0, 01 = +1, 10 = -1

    Args:
        weights: [M, K] int8 with values in {-1, 0, 1}

    Returns:
        packed: [M, K // 4] uint8
    """
    M, K = weights.shape
    assert K % 4 == 0, f"K must be divisible by 4, got {K}"

    # Encode: +1 → 0b01, -1 → 0b10, 0 → 0b00
    encoded = mx.where(
        weights == 1,
        mx.array(1, dtype=mx.uint8),
        mx.where(weights == -1, mx.array(2, dtype=mx.uint8), mx.array(0, dtype=mx.uint8)),
    )
    encoded = encoded.reshape(M, K // 4, 4)
    packed = (
        encoded[:, :, 0]
        | (encoded[:, :, 1] << 2)
        | (encoded[:, :, 2] << 4)
        | (encoded[:, :, 3] << 6)
    )
    return packed.astype(mx.uint8)


def unpack_ternary(packed: mx.array, K: int) -> mx.array:
    """Unpack 2-bit packed uint8 to ternary float16.

    Args:
        packed: [M, K // 4] uint8
        K: original inner dimension

    Returns:
        weights: [M, K] float16 with values in {-1.0, 0.0, +1.0}
    """
    M = packed.shape[0]
    w0 = packed & 0x03
    w1 = (packed >> 2) & 0x03
    w2 = (packed >> 4) & 0x03
    w3 = (packed >> 6) & 0x03

    # Stack and reshape: [M, K//4, 4] → [M, K]
    unpacked = mx.stack([w0, w1, w2, w3], axis=-1).reshape(M, K)

    # Decode: 01 → +1, 10 → -1, 00 → 0
    # Formula: value = (w & 1) - (w >> 1)
    decoded = (unpacked & 1).astype(mx.float16) - (unpacked >> 1).astype(mx.float16)
    return decoded


def compute_ternary_stats(weight: mx.array) -> dict:
    """Compute statistics about a ternary quantization."""
    scale = mx.mean(mx.abs(weight))
    normalized = weight / mx.maximum(scale, mx.array(1e-5))
    ternary = mx.clip(mx.round(normalized), -1, 1).astype(mx.int8)

    total = ternary.size
    n_zero = mx.sum(ternary == 0).item()
    n_pos = mx.sum(ternary == 1).item()
    n_neg = mx.sum(ternary == -1).item()

    return {
        "scale": scale.item(),
        "sparsity": n_zero / total,
        "pos_frac": n_pos / total,
        "neg_frac": n_neg / total,
        "zero_frac": n_zero / total,
    }

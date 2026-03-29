"""Custom Metal compute shader for ternary matrix-vector multiply."""

import mlx.core as mx

# Metal kernel source for ternary GEMV (decode path).
# Weights are packed as 2 bits each: 00=0, 01=+1, 10=-1.
# Matmul becomes conditional add/subtract — no multiplication on weights.
TERNARY_MATVEC_SOURCE = """
    // Simple approach: one thread per output row
    // This is memory-bandwidth-bound anyway, so thread-level parallelism
    // across rows is more important than within-row reduction.
    uint row = thread_position_in_grid.x;
    uint M_val = packed_w_shape[0];
    uint packed_K = packed_w_shape[1];

    if (row >= M_val) return;

    float sum = 0.0f;
    uint w_base = row * packed_K;

    for (uint i = 0; i < packed_K; i++) {
        uchar pack = packed_w[w_base + i];
        uint base_k = i * 4;

        uint w0 = pack & 0x3;
        uint w1 = (pack >> 2) & 0x3;
        uint w2 = (pack >> 4) & 0x3;
        uint w3 = (pack >> 6) & 0x3;

        float s0 = float(w0 & 1) - float(w0 >> 1);
        float s1 = float(w1 & 1) - float(w1 >> 1);
        float s2 = float(w2 & 1) - float(w2 >> 1);
        float s3 = float(w3 & 1) - float(w3 >> 1);

        sum += s0 * float(inp[base_k])
             + s1 * float(inp[base_k + 1])
             + s2 * float(inp[base_k + 2])
             + s3 * float(inp[base_k + 3]);
    }

    output[row] = half(sum * float(scale[0]));
"""

_kernel_cache = {}


def _get_kernel():
    """Get or create the cached Metal kernel."""
    if "matvec" not in _kernel_cache:
        _kernel_cache["matvec"] = mx.fast.metal_kernel(
            name="ternary_matvec",
            input_names=["packed_w", "inp", "scale"],
            output_names=["output"],
            source=TERNARY_MATVEC_SOURCE,
        )
    return _kernel_cache["matvec"]


def ternary_matvec(
    packed_weights: mx.array,
    input_vec: mx.array,
    scale: mx.array,
) -> mx.array:
    """Ternary matrix-vector multiply using custom Metal kernel.

    Computes: output = (unpack(packed_weights) @ input_vec) * scale

    Args:
        packed_weights: [M, K//4] uint8 — packed ternary weights
        input_vec: [K] float16 — input activation vector
        scale: [1] float16 — per-tensor weight scale

    Returns:
        output: [M] float16
    """
    M = packed_weights.shape[0]
    # One thread per output row, dispatched across threadgroups
    tg_size = min(256, M)
    grid_size = (M + tg_size - 1) // tg_size  # number of threadgroups

    kernel = _get_kernel()
    outputs = kernel(
        inputs=[packed_weights, input_vec, scale],
        grid=(grid_size * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(M,)],
        output_dtypes=[mx.float16],
        stream=mx.gpu,
    )
    return outputs[0]


def ternary_matvec_fallback(
    packed_weights: mx.array,
    input_vec: mx.array,
    scale: mx.array,
    K: int,
) -> mx.array:
    """Fallback ternary GEMV using standard MLX ops (no custom kernel).

    Used when the Metal kernel is not available or for debugging.
    """
    from onebit.quant import unpack_ternary

    weights = unpack_ternary(packed_weights, K)  # [M, K] float16
    out = weights @ input_vec  # [M]
    return out * scale

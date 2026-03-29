"""BitLinear: ternary linear layer for 1-bit LLM inference."""

import mlx.core as mx
import mlx.nn as nn

from onebit.quant import quantize_to_ternary, unpack_ternary


class BitLinear(nn.Module):
    """Linear layer with ternary weights {-1, 0, +1}.

    Stores weights packed at 2 bits per value. Uses a custom Metal kernel
    for single-token decode (GEMV) and unpacked MLX matmul for prefill (GEMM).

    Attributes:
        packed_weights: [out_features, in_features // 4] uint8
        weight_scale:   [1] float16 — per-tensor scale
        bias:           [out_features] float16 (optional)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_metal_kernel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_metal_kernel = use_metal_kernel

        self.packed_weights = mx.zeros(
            (out_features, in_features // 4), dtype=mx.uint8
        )
        self.weight_scale = mx.ones((1,), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        For single-token input (decode): uses custom Metal GEMV kernel.
        For multi-token input (prefill): unpacks weights and uses MLX matmul.
        """
        orig_shape = x.shape
        # x: [..., in_features]
        if x.ndim > 2:
            x = x.reshape(-1, self.in_features)

        B = x.shape[0]

        if B == 1 and self.use_metal_kernel:
            out = self._decode_forward(x.squeeze(0))
            out = out.reshape(1, -1)
        else:
            out = self._prefill_forward(x)

        # Reshape to match input batch dims
        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], self.out_features)

        if hasattr(self, "bias"):
            out = out + self.bias

        return out

    def _decode_forward(self, x: mx.array) -> mx.array:
        """Single-token decode using custom Metal kernel."""
        try:
            from onebit.kernels import ternary_matvec

            result = ternary_matvec(
                self.packed_weights,
                x.astype(mx.float16),
                self.weight_scale,
            )
            return result
        except Exception:
            return self._prefill_forward(x.reshape(1, -1)).squeeze(0)

    def _prefill_forward(self, x: mx.array) -> mx.array:
        """Multi-token prefill using standard MLX matmul."""
        weights = unpack_ternary(self.packed_weights, self.in_features)
        out = x @ weights.T
        # weight_scale is [out_features] (per-row) or [1] (per-tensor)
        return out * self.weight_scale

    @classmethod
    def from_linear(cls, linear: nn.Linear, use_metal_kernel: bool = True) -> "BitLinear":
        """Convert a standard nn.Linear to BitLinear with ternary weights."""
        has_bias = hasattr(linear, "bias") and linear.bias is not None
        out_features, in_features = linear.weight.shape

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            use_metal_kernel=use_metal_kernel,
        )

        packed, scale = quantize_to_ternary(linear.weight)
        layer.packed_weights = packed
        layer.weight_scale = scale

        if has_bias:
            layer.bias = linear.bias

        return layer

    @classmethod
    def from_packed(
        cls,
        packed_weights: mx.array,
        weight_scale: mx.array,
        in_features: int,
        bias: mx.array | None = None,
        use_metal_kernel: bool = True,
    ) -> "BitLinear":
        """Create BitLinear from pre-packed ternary weights."""
        out_features = packed_weights.shape[0]
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            use_metal_kernel=use_metal_kernel,
        )
        layer.packed_weights = packed_weights
        layer.weight_scale = weight_scale
        if bias is not None:
            layer.bias = bias
        return layer

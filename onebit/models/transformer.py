"""Generic transformer model supporting Llama, Qwen, Phi, and BitNet architectures."""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from onebit.layers import BitLinear
from onebit.models.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rms).astype(dtype) * self.weight


class KVCache:
    """Simple key-value cache for autoregressive generation."""

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    @property
    def offset(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values


def _make_linear(in_f: int, out_f: int, bias: bool, ternary: bool) -> nn.Module:
    """Create either a BitLinear or standard nn.Linear."""
    if ternary:
        return BitLinear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, ternary: bool = True):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.n_rep = self.num_heads // self.num_kv_heads

        bias = config.attention_bias
        hidden = config.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = _make_linear(hidden, q_dim, bias, ternary)
        self.k_proj = _make_linear(hidden, kv_dim, bias, ternary)
        self.v_proj = _make_linear(hidden, kv_dim, bias, ternary)
        self.o_proj = _make_linear(q_dim, hidden, bias, ternary)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply rotary position embeddings
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV cache
        if cache is not None:
            k, v = cache.update(k, v)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP used by Llama, Qwen, Phi, BitNet."""

    def __init__(self, config: ModelConfig, ternary: bool = True):
        super().__init__()
        hidden = config.hidden_size
        inter = config.intermediate_size
        bias = config.mlp_bias

        self.gate_proj = _make_linear(hidden, inter, bias, ternary)
        self.up_proj = _make_linear(hidden, inter, bias, ternary)
        self.down_proj = _make_linear(inter, hidden, bias, ternary)

        if config.hidden_act == "silu":
            self.act = nn.SiLU()
        elif config.hidden_act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, ternary: bool = True):
        super().__init__()
        self.self_attn = Attention(config, ternary=ternary)
        self.mlp = MLP(config, ternary=ternary)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class TransformerModelInner(nn.Module):
    """Inner model (without lm_head) — matches HF 'model.*' weight namespace."""

    def __init__(self, config: ModelConfig, ternary: bool = True):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config, ternary=ternary)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class TransformerModel(nn.Module):
    """Full causal LM transformer with ternary weights.

    Weight naming matches HuggingFace conventions:
        model.embed_tokens.weight
        model.layers.0.self_attn.q_proj.packed_weights
        model.layers.0.self_attn.q_proj.weight_scale
        ...
        model.norm.weight
        lm_head.weight
    """

    def __init__(self, config: ModelConfig, ternary: bool = True):
        super().__init__()
        self.config = config
        self.model = TransformerModelInner(config, ternary=ternary)
        # lm_head stays in FP16 for quality
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        h = self.model.embed_tokens(input_ids)

        # Causal mask for prefill (multi-token input)
        mask = None
        L = input_ids.shape[1]
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(h.dtype)

        for i, layer in enumerate(self.model.layers):
            h = layer(h, mask=mask, cache=cache[i] if cache else None)

        h = self.model.norm(h)
        return self.lm_head(h)

    def make_cache(self) -> list[KVCache]:
        """Create a fresh KV cache for generation."""
        return [KVCache() for _ in range(self.config.num_hidden_layers)]

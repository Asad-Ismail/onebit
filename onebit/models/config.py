"""Model configuration — maps HuggingFace configs to our transformer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Unified config for all supported architectures (Llama, Qwen, Phi, BitNet)."""

    model_type: str = "llama"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 5504
    vocab_size: int = 32064
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False

    # Phi-specific: fused qkv and gate_up projections in HF weights
    fused_qkv: bool = False
    fused_gate_up: bool = False

    # Onebit metadata
    is_ternary: bool = False
    base_model: str = ""
    quantize_lm_head: bool = False

    # Rope scaling
    rope_scaling: dict = field(default_factory=dict)

    @classmethod
    def from_hf_config(cls, config_path: str | Path) -> ModelConfig:
        """Load from a HuggingFace config.json file."""
        with open(config_path) as f:
            hf = json.load(f)

        model_type = hf.get("model_type", "llama")

        # Detect fused projections (Phi-3/4)
        fused_qkv = model_type in ("phi3", "phi")
        fused_gate_up = model_type in ("phi3", "phi")

        # Handle num_key_value_heads defaulting to num_attention_heads
        n_heads = hf.get("num_attention_heads", 32)
        n_kv_heads = hf.get("num_key_value_heads", n_heads)

        # Qwen2 defaults attention_bias to True when not specified
        default_attn_bias = model_type in ("qwen2", "qwen2_moe")
        attn_bias = hf.get("attention_bias", default_attn_bias)
        # Treat None as the default for this model type
        if attn_bias is None:
            attn_bias = default_attn_bias

        return cls(
            model_type=model_type,
            hidden_size=hf.get("hidden_size", 2048),
            num_hidden_layers=hf.get("num_hidden_layers", 24),
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            intermediate_size=hf.get("intermediate_size", 5504),
            vocab_size=hf.get("vocab_size", 32064),
            rms_norm_eps=hf.get("rms_norm_eps", 1e-5),
            rope_theta=hf.get("rope_theta", 10000.0),
            max_position_embeddings=hf.get("max_position_embeddings", 4096),
            hidden_act=hf.get("hidden_act", hf.get("hidden_activation", "silu")),
            attention_bias=attn_bias,
            mlp_bias=hf.get("mlp_bias", False) or False,
            tie_word_embeddings=hf.get("tie_word_embeddings", False),
            fused_qkv=fused_qkv,
            fused_gate_up=fused_gate_up,
            is_ternary=hf.get("quantization_config", {}).get("quant_method") == "bitnet"
            or hf.get("onebit", {}).get("quantization") == "ternary_1.58bit",
            base_model=hf.get("onebit", {}).get("base_model", ""),
            rope_scaling=hf.get("rope_scaling", {}),
        )

    def to_dict(self) -> dict:
        """Serialize to dict for saving."""
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "tie_word_embeddings": self.tie_word_embeddings,
            "fused_qkv": self.fused_qkv,
            "fused_gate_up": self.fused_gate_up,
            "onebit": {
                "quantization": "ternary_1.58bit",
                "base_model": self.base_model,
            },
        }

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

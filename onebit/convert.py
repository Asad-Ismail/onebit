"""Convert any HuggingFace model to 1.58-bit ternary format."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlx.core as mx

from onebit.engine import _download_hf, _load_safetensors
from onebit.models.config import ModelConfig
from onebit.quant import quantize_to_ternary

logger = logging.getLogger(__name__)


def convert_model(
    hf_model_id: str,
    output_dir: str,
    skip_lm_head: bool = True,
    skip_embeddings: bool = True,
) -> Path:
    """Convert a HuggingFace model to onebit ternary format.

    Args:
        hf_model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
        output_dir: Directory to save the converted model
        skip_lm_head: Keep lm_head in FP16 (recommended for quality)
        skip_embeddings: Keep embeddings in FP16

    Returns:
        Path to the output directory
    """
    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download model
    logger.info(f"Downloading {hf_model_id}...")
    model_dir = _download_hf(hf_model_id)

    # Load config
    config = ModelConfig.from_hf_config(model_dir / "config.json")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    # Load weights
    logger.info("Loading weights...")
    raw_weights = _load_safetensors(model_dir)

    # Quantize
    logger.info("Quantizing to 1.58-bit ternary...")
    t0 = time.time()
    converted = {}
    total_params = 0
    quantized_params = 0

    skip_prefixes = set()
    if skip_embeddings:
        skip_prefixes.add("model.embed_tokens.")
    if skip_lm_head:
        skip_prefixes.add("lm_head.")

    for name, weight in raw_weights.items():
        total_params += weight.size

        # Check if this weight should be skipped
        should_skip = any(name.startswith(p) for p in skip_prefixes)
        is_norm = "layernorm" in name.lower() or "norm" in name.lower()
        is_weight = name.endswith(".weight") and not is_norm

        if is_weight and not should_skip and weight.ndim == 2:
            # Handle fused QKV projections
            if config.fused_qkv and "qkv_proj.weight" in name:
                q_dim = config.num_attention_heads * config.head_dim
                kv_dim = config.num_key_value_heads * config.head_dim
                q_w, k_w, v_w = mx.split(weight, [q_dim, q_dim + kv_dim], axis=0)
                base = name.replace("qkv_proj.weight", "")
                for proj_name, proj_w in [("q_proj", q_w), ("k_proj", k_w), ("v_proj", v_w)]:
                    packed, scale = quantize_to_ternary(proj_w)
                    converted[f"{base}{proj_name}.packed_weights"] = packed
                    converted[f"{base}{proj_name}.weight_scale"] = scale
                    quantized_params += proj_w.size
                continue

            # Handle fused gate_up projections
            if config.fused_gate_up and "gate_up_proj.weight" in name:
                gate_w, up_w = mx.split(weight, 2, axis=0)
                base = name.replace("gate_up_proj.weight", "")
                for proj_name, proj_w in [("gate_proj", gate_w), ("up_proj", up_w)]:
                    packed, scale = quantize_to_ternary(proj_w)
                    converted[f"{base}{proj_name}.packed_weights"] = packed
                    converted[f"{base}{proj_name}.weight_scale"] = scale
                    quantized_params += proj_w.size
                continue

            # Standard linear layer: quantize
            base = name[: -len(".weight")]
            packed, scale = quantize_to_ternary(weight)
            converted[f"{base}.packed_weights"] = packed
            converted[f"{base}.weight_scale"] = scale
            quantized_params += weight.size
        else:
            # Keep as-is (norms, biases, embeddings, lm_head)
            converted[name] = weight

    dt = time.time() - t0
    logger.info(
        f"Quantized {quantized_params:,} / {total_params:,} params "
        f"({quantized_params / total_params * 100:.1f}%) in {dt:.1f}s"
    )

    # Save weights
    logger.info("Saving ternary weights...")
    mx.save_safetensors(str(output_path / "model.safetensors"), converted)

    # Save config
    config_dict = config.to_dict()
    config_dict["onebit"] = {
        "quantization": "ternary_1.58bit",
        "base_model": hf_model_id,
        "quantized_params": quantized_params,
        "total_params": total_params,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save onebit marker
    with open(output_path / "onebit_config.json", "w") as f:
        json.dump(
            {
                "quantization": "ternary_1.58bit",
                "version": "0.1.0",
                "base_model": hf_model_id,
            },
            f,
            indent=2,
        )

    # Compute size savings
    original_size_mb = total_params * 2 / (1024 * 1024)  # FP16
    ternary_size = sum(v.nbytes for v in converted.values())
    ternary_size_mb = ternary_size / (1024 * 1024)

    logger.info(
        f"Size: {original_size_mb:.0f} MB (FP16) -> {ternary_size_mb:.0f} MB (ternary) "
        f"({original_size_mb / ternary_size_mb:.1f}x compression)"
    )

    return output_path

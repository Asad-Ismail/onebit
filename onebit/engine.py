"""Model loading, weight mapping, and management."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from onebit.layers import BitLinear
from onebit.models.config import ModelConfig
from onebit.models.transformer import TransformerModel
from onebit.quant import quantize_to_ternary

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "onebit"


def load_model(
    model_name_or_path: str,
    use_metal_kernel: bool = True,
) -> tuple[TransformerModel, "AutoTokenizer"]:
    """Load a model by registry name, HF repo ID, or local path.

    If the model is not yet quantized to ternary, it will be quantized
    on-the-fly and cached locally.

    Returns:
        (model, tokenizer) tuple ready for generation.
    """
    from onebit.models.registry import MODELS

    # Resolve model path
    if model_name_or_path in MODELS:
        info = MODELS[model_name_or_path]
        hf_repo = info["hf_repo"]
        cache_name = model_name_or_path
    elif Path(model_name_or_path).is_dir():
        return _load_local(Path(model_name_or_path), use_metal_kernel)
    else:
        hf_repo = model_name_or_path
        cache_name = hf_repo.replace("/", "--")

    # Check for cached ternary model
    cached_path = CACHE_DIR / cache_name
    if cached_path.exists() and (cached_path / "config.json").exists():
        logger.info(f"Loading cached ternary model from {cached_path}")
        return _load_local(cached_path, use_metal_kernel)

    # Download and convert
    logger.info(f"Downloading {hf_repo} from HuggingFace...")
    local_path = _download_hf_model(hf_repo)

    logger.info("Quantizing to 1.58-bit ternary...")
    t0 = time.time()
    model, tokenizer = _load_and_quantize(local_path, use_metal_kernel)
    dt = time.time() - t0
    logger.info(f"Quantization complete in {dt:.1f}s")

    # Cache the ternary model
    _save_ternary_model(model, tokenizer, local_path, cached_path)
    logger.info(f"Cached ternary model at {cached_path}")

    return model, tokenizer


def _download_hf_model(repo_id: str) -> Path:
    """Download model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    return Path(local_dir)


def _load_local(
    model_dir: Path, use_metal_kernel: bool = True
) -> tuple[TransformerModel, "AutoTokenizer"]:
    """Load a model from a local directory (either FP16 or pre-quantized ternary)."""
    from transformers import AutoTokenizer

    config = ModelConfig.from_hf_config(model_dir / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Check if this is already a ternary model
    onebit_config = model_dir / "onebit_config.json"
    is_ternary = onebit_config.exists()

    if is_ternary:
        model = TransformerModel(config, ternary=True)
        _load_ternary_weights(model, model_dir, config)
    else:
        # Load FP16 and quantize on the fly
        model = TransformerModel(config, ternary=True)
        _load_and_quantize_weights(model, model_dir, config, use_metal_kernel)

    mx.eval(model.parameters())
    return model, tokenizer


def _load_and_quantize(
    model_dir: Path, use_metal_kernel: bool = True
) -> tuple[TransformerModel, "AutoTokenizer"]:
    """Load FP16 model and quantize all linear layers to ternary."""
    from transformers import AutoTokenizer

    config = ModelConfig.from_hf_config(model_dir / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    model = TransformerModel(config, ternary=True)
    _load_and_quantize_weights(model, model_dir, config, use_metal_kernel)
    mx.eval(model.parameters())

    return model, tokenizer


def _load_safetensors(model_dir: Path) -> dict[str, mx.array]:
    """Load all safetensors files from a directory into a flat dict."""
    import glob

    weights = {}
    for path in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        w = mx.load(path)
        weights.update(w)
    return weights


def _load_and_quantize_weights(
    model: TransformerModel,
    model_dir: Path,
    config: ModelConfig,
    use_metal_kernel: bool = True,
) -> None:
    """Load FP16 weights, quantize linear layers to ternary, and assign to model."""
    raw_weights = _load_safetensors(model_dir)

    # Build the weight mapping
    mapped = {}
    n_layers = config.num_hidden_layers

    # Embedding and norm (keep FP16)
    mapped["model.embed_tokens.weight"] = raw_weights.get("model.embed_tokens.weight")

    if "model.norm.weight" in raw_weights:
        mapped["model.norm.weight"] = raw_weights["model.norm.weight"]

    # LM head (keep FP16)
    if "lm_head.weight" in raw_weights:
        mapped["lm_head.weight"] = raw_weights["lm_head.weight"]
    elif config.tie_word_embeddings:
        mapped["lm_head.weight"] = raw_weights["model.embed_tokens.weight"]

    # Transformer layers
    for i in range(n_layers):
        pfx = f"model.layers.{i}"

        # Layer norms (keep FP16)
        mapped[f"{pfx}.input_layernorm.weight"] = raw_weights[f"{pfx}.input_layernorm.weight"]
        mapped[f"{pfx}.post_attention_layernorm.weight"] = raw_weights[
            f"{pfx}.post_attention_layernorm.weight"
        ]

        # Attention projections
        if config.fused_qkv and f"{pfx}.self_attn.qkv_proj.weight" in raw_weights:
            qkv_w = raw_weights[f"{pfx}.self_attn.qkv_proj.weight"]
            q_dim = config.num_attention_heads * config.head_dim
            kv_dim = config.num_key_value_heads * config.head_dim
            q_w, k_w, v_w = mx.split(qkv_w, [q_dim, q_dim + kv_dim], axis=0)
            _quantize_and_map(mapped, f"{pfx}.self_attn.q_proj", q_w)
            _quantize_and_map(mapped, f"{pfx}.self_attn.k_proj", k_w)
            _quantize_and_map(mapped, f"{pfx}.self_attn.v_proj", v_w)

            # Handle fused bias if present
            bias_key = f"{pfx}.self_attn.qkv_proj.bias"
            if bias_key in raw_weights:
                qkv_b = raw_weights[bias_key]
                q_b, k_b, v_b = mx.split(qkv_b, [q_dim, q_dim + kv_dim], axis=0)
                mapped[f"{pfx}.self_attn.q_proj.bias"] = q_b
                mapped[f"{pfx}.self_attn.k_proj.bias"] = k_b
                mapped[f"{pfx}.self_attn.v_proj.bias"] = v_b
        else:
            for proj in ("q_proj", "k_proj", "v_proj"):
                key = f"{pfx}.self_attn.{proj}.weight"
                if key in raw_weights:
                    _quantize_and_map(mapped, f"{pfx}.self_attn.{proj}", raw_weights[key])
                bias_key = f"{pfx}.self_attn.{proj}.bias"
                if bias_key in raw_weights:
                    mapped[bias_key] = raw_weights[bias_key]

        # Output projection
        o_key = f"{pfx}.self_attn.o_proj.weight"
        if o_key in raw_weights:
            _quantize_and_map(mapped, f"{pfx}.self_attn.o_proj", raw_weights[o_key])
        o_bias = f"{pfx}.self_attn.o_proj.bias"
        if o_bias in raw_weights:
            mapped[o_bias] = raw_weights[o_bias]

        # MLP
        if config.fused_gate_up and f"{pfx}.mlp.gate_up_proj.weight" in raw_weights:
            gu_w = raw_weights[f"{pfx}.mlp.gate_up_proj.weight"]
            gate_w, up_w = mx.split(gu_w, 2, axis=0)
            _quantize_and_map(mapped, f"{pfx}.mlp.gate_proj", gate_w)
            _quantize_and_map(mapped, f"{pfx}.mlp.up_proj", up_w)
        else:
            for proj in ("gate_proj", "up_proj"):
                key = f"{pfx}.mlp.{proj}.weight"
                if key in raw_weights:
                    _quantize_and_map(mapped, f"{pfx}.mlp.{proj}", raw_weights[key])

        down_key = f"{pfx}.mlp.down_proj.weight"
        if down_key in raw_weights:
            _quantize_and_map(mapped, f"{pfx}.mlp.down_proj", raw_weights[down_key])

    # Assign weights to model (strict=False to handle unexpected extra params)
    model.load_weights(list(mapped.items()), strict=False)


def _quantize_and_map(mapped: dict, prefix: str, weight: mx.array) -> None:
    """Quantize a weight to ternary and add packed_weights + weight_scale to the mapping."""
    packed, scale = quantize_to_ternary(weight)
    mapped[f"{prefix}.packed_weights"] = packed
    mapped[f"{prefix}.weight_scale"] = scale


def _load_ternary_weights(
    model: TransformerModel,
    model_dir: Path,
    config: ModelConfig,
) -> None:
    """Load pre-quantized ternary weights from safetensors."""
    weights = _load_safetensors(model_dir)
    model.load_weights(list(weights.items()), strict=False)


def _save_ternary_model(
    model: TransformerModel,
    tokenizer: "AutoTokenizer",
    original_dir: Path,
    output_dir: Path,
) -> None:
    """Save a ternary model to disk for fast loading next time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = model.config.to_dict()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save onebit marker
    with open(output_dir / "onebit_config.json", "w") as f:
        json.dump({"quantization": "ternary_1.58bit", "version": "0.1.0"}, f)

    # Save weights
    weights = dict(model.parameters())
    flat_weights = {}
    _flatten_params(weights, "", flat_weights)
    mx.save_safetensors(str(output_dir / "model.safetensors"), flat_weights)

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"Saved ternary model to {output_dir}")


def _flatten_params(params, prefix: str, out: dict) -> None:
    """Flatten nested parameter dict to dot-separated keys."""
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            _flatten_params(v, new_prefix, out)
    elif isinstance(params, list):
        for i, v in enumerate(params):
            _flatten_params(v, f"{prefix}.{i}", out)
    elif isinstance(params, mx.array):
        out[prefix] = params


def get_model_memory_mb(model: TransformerModel) -> float:
    """Estimate model memory usage in MB."""
    total_bytes = 0
    for _, p in model.parameters().items() if isinstance(model.parameters(), dict) else []:
        total_bytes += p.nbytes
    # Fallback: walk parameters
    params = dict(_iter_params(model))
    total_bytes = sum(p.nbytes for p in params.values())
    return total_bytes / (1024 * 1024)


def _iter_params(module, prefix=""):
    """Iterate over all parameters in a module."""
    if isinstance(module, mx.array):
        yield prefix, module
    elif isinstance(module, nn.Module):
        for k, v in vars(module).items():
            yield from _iter_params(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(module, dict):
        for k, v in module.items():
            yield from _iter_params(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(module, list):
        for i, v in enumerate(module):
            yield from _iter_params(v, f"{prefix}.{i}" if prefix else str(i))

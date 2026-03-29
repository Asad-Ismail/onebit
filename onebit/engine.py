"""Model loading, weight mapping, and management.

Supports two quantization paths:
1. mlx-lm 2-bit: Proven quality, uses MLX-LM's quantization engine (default)
2. Native ternary: For models trained with ternary weights (BitNet b1.58)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.utils

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "onebit"


def load_model(
    model_name_or_path: str,
    bits: int = 4,
    use_metal_kernel: bool = True,
):
    """Load a model by registry name, HF repo ID, or local path.

    For non-ternary models, uses mlx-lm's quantization for quality.
    For natively-trained ternary models, uses our custom ternary path.

    Args:
        model_name_or_path: Registry name, HF repo, or local dir
        bits: Quantization bits (2 for quality, or 158 for native ternary)
        use_metal_kernel: Use custom Metal kernel for ternary decode

    Returns:
        (model, tokenizer) tuple ready for generation.
    """
    from onebit.models.registry import MODELS

    # Resolve model
    if model_name_or_path in MODELS:
        info = MODELS[model_name_or_path]
        hf_repo = info["hf_repo"]
        cache_name = model_name_or_path
        is_native_ternary = info.get("native_ternary", False)
    elif Path(model_name_or_path).is_dir():
        return _load_local_dir(Path(model_name_or_path))
    else:
        hf_repo = model_name_or_path
        cache_name = hf_repo.replace("/", "--")
        is_native_ternary = False

    # Check cache
    cached_path = CACHE_DIR / cache_name
    if cached_path.exists() and (cached_path / "config.json").exists():
        logger.info(f"Loading cached model from {cached_path}")
        return _load_local_dir(cached_path)

    # Download and quantize
    if is_native_ternary:
        return _load_native_ternary(hf_repo, cached_path, use_metal_kernel)
    else:
        return _load_and_quantize_mlxlm(hf_repo, cached_path, bits)


def _load_and_quantize_mlxlm(hf_repo: str, cache_path: Path, bits: int = 2):
    """Load and quantize using mlx-lm's proven quantization."""
    import mlx_lm

    logger.info(f"Downloading {hf_repo}...")

    # Use mlx-lm to load the full-precision model
    logger.info("Loading model with mlx-lm...")
    model, tokenizer = mlx_lm.load(hf_repo)

    # Quantize with mlx-lm
    logger.info(f"Quantizing to {bits}-bit with mlx-lm...")
    t0 = time.time()

    # mlx-lm quantize: convert linear layers to QuantizedLinear
    from mlx_lm.utils import quantize_model as _mlx_quantize

    q_config = {"group_size": 64, "bits": bits}
    model, q_config = _mlx_quantize(model, q_config, group_size=64, bits=bits)
    mx.eval(model.parameters())

    dt = time.time() - t0
    logger.info(f"Quantization complete in {dt:.1f}s")

    # Save using mlx-lm's save which handles QuantizedLinear correctly
    cache_path.mkdir(parents=True, exist_ok=True)
    from mlx_lm.utils import save_model as _mlx_save
    _mlx_save(str(cache_path), model)
    tokenizer.save_pretrained(str(cache_path))
    # Add quantization config to config.json
    config_file = cache_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        cfg.setdefault("quantization", {"group_size": 64, "bits": bits})
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2)
    logger.info(f"Cached at {cache_path}")

    return model, tokenizer


def _load_native_ternary(hf_repo: str, cache_path: Path, use_metal_kernel: bool):
    """Load a natively-trained ternary model (BitNet b1.58)."""
    from onebit.models.config import ModelConfig
    from onebit.models.transformer import TransformerModel
    from transformers import AutoTokenizer

    logger.info(f"Downloading native ternary model {hf_repo}...")
    local_path = _download_hf(hf_repo)

    config = ModelConfig.from_hf_config(local_path / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(local_path), trust_remote_code=True)

    model = TransformerModel(config, ternary=True)
    _load_and_quantize_weights_ternary(model, local_path, config)
    mx.eval(model.parameters())

    # Cache
    cache_path.mkdir(parents=True, exist_ok=True)
    _save_ternary_model(model, tokenizer, local_path, cache_path)

    return model, tokenizer


def _load_local_dir(model_dir: Path):
    """Load from a local directory (auto-detect format)."""
    # Check if it's an mlx-lm quantized model
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    # Check for onebit ternary marker
    onebit_config = model_dir / "onebit_config.json"
    if onebit_config.exists():
        with open(onebit_config) as f:
            ob = json.load(f)
        if ob.get("format") == "ternary":
            return _load_ternary_local(model_dir)

    # Default: load with mlx-lm (handles both quantized and FP16)
    import mlx_lm
    return mlx_lm.load(str(model_dir))


def _load_ternary_local(model_dir: Path):
    """Load a native ternary model from local cache."""
    from onebit.models.config import ModelConfig
    from onebit.models.transformer import TransformerModel
    from transformers import AutoTokenizer

    config = ModelConfig.from_hf_config(model_dir / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    model = TransformerModel(config, ternary=True)
    weights = _load_safetensors(model_dir)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    return model, tokenizer


def _download_hf(repo_id: str) -> Path:
    """Download model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    return Path(local_dir)


def _load_safetensors(model_dir: Path) -> dict[str, mx.array]:
    """Load all safetensors files from a directory."""
    import glob

    weights = {}
    for path in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        w = mx.load(path)
        weights.update(w)
    return weights


def _load_and_quantize_weights_ternary(model, model_dir, config):
    """Load FP16 weights and quantize linear layers to ternary."""
    from onebit.quant import quantize_to_ternary

    raw_weights = _load_safetensors(model_dir)
    mapped = {}
    n_layers = config.num_hidden_layers

    # Embedding and norms (keep FP16)
    if "model.embed_tokens.weight" in raw_weights:
        mapped["model.embed_tokens.weight"] = raw_weights["model.embed_tokens.weight"]
    if "model.norm.weight" in raw_weights:
        mapped["model.norm.weight"] = raw_weights["model.norm.weight"]
    if "lm_head.weight" in raw_weights:
        mapped["lm_head.weight"] = raw_weights["lm_head.weight"]
    elif config.tie_word_embeddings:
        mapped["lm_head.weight"] = raw_weights.get("model.embed_tokens.weight")

    for i in range(n_layers):
        pfx = f"model.layers.{i}"
        for key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            full_key = f"{pfx}.{key}"
            if full_key in raw_weights:
                mapped[full_key] = raw_weights[full_key]

        # Quantize attention and MLP weights
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            w_key = f"{pfx}.{proj}.weight"
            if w_key in raw_weights:
                packed, scale = quantize_to_ternary(raw_weights[w_key])
                mapped[f"{pfx}.{proj}.packed_weights"] = packed
                mapped[f"{pfx}.{proj}.weight_scale"] = scale

            # Pass through biases
            b_key = f"{pfx}.{proj}.bias"
            if b_key in raw_weights:
                mapped[b_key] = raw_weights[b_key]

    model.load_weights(list(mapped.items()), strict=False)


def _save_mlxlm_model(model, tokenizer, base_model, output_dir, bits):
    """Save mlx-lm quantized model."""
    import mlx_lm

    # Save weights
    weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))

    # Copy config and add our metadata
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(base_model, "config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        config = {}

    config["quantization_config"] = {"bits": bits, "group_size": 64, "quant_method": "mlx"}
    config["onebit"] = {"base_model": base_model, "bits": bits}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _save_ternary_model(model, tokenizer, original_dir, output_dir):
    """Save a native ternary model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "onebit_config.json", "w") as f:
        json.dump({"format": "ternary", "version": "0.1.0"}, f)

    config_dict = model.config.to_dict()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def get_model_size_mb(model) -> float:
    """Estimate model memory in MB."""
    total = 0
    for _, v in mlx.utils.tree_flatten(model.parameters()):
        total += v.nbytes
    return total / (1024 * 1024)

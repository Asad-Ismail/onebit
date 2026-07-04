"""Model loading, weight mapping, and management.

Two load paths:
1. mlx-lm 4-bit (default): download an FP16 checkpoint and quantize it with
   MLX-LM's proven engine. This runs for every registry model and any HF repo.
2. onebit ternary (experimental): load a model produced by `onebit convert`
   (marked with onebit_config.json) into the custom transformer + Metal kernel.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "onebit"


def load_model(model_name_or_path: str, bits: int = 4):
    """Load a model by registry name, HF repo ID, or local path.

    Registry names and HF repos are quantized with mlx-lm on first run and
    cached under ``~/.cache/onebit``. Local directories are auto-detected: an
    onebit-converted ternary model loads via the custom path, anything else via
    mlx-lm.

    Args:
        model_name_or_path: Registry name, HF repo, or local dir.
        bits: Bit width for the mlx-lm quantization path (default 4).

    Returns:
        (model, tokenizer) tuple ready for generation.
    """
    from onebit.models.registry import MODELS

    if model_name_or_path in MODELS:
        hf_repo = MODELS[model_name_or_path]["hf_repo"]
        cache_name = model_name_or_path
    elif Path(model_name_or_path).is_dir():
        return _load_local_dir(Path(model_name_or_path))
    else:
        hf_repo = model_name_or_path
        cache_name = hf_repo.replace("/", "--")

    cached_path = CACHE_DIR / cache_name
    if cached_path.exists() and (cached_path / "config.json").exists():
        logger.info(f"Loading cached model from {cached_path}")
        return _load_local_dir(cached_path)

    return _load_and_quantize_mlxlm(hf_repo, cached_path, bits)


def _load_and_quantize_mlxlm(hf_repo: str, cache_path: Path, bits: int = 4):
    """Download an FP16 checkpoint and quantize it with mlx-lm."""
    import mlx_lm
    from mlx_lm.utils import quantize_model as _mlx_quantize
    from mlx_lm.utils import save_model as _mlx_save

    logger.info(f"Downloading and loading {hf_repo} with mlx-lm...")
    model, tokenizer = mlx_lm.load(hf_repo)

    logger.info(f"Quantizing to {bits}-bit with mlx-lm...")
    t0 = time.time()
    q_config = {"group_size": 64, "bits": bits}
    model, q_config = _mlx_quantize(model, q_config, group_size=64, bits=bits)
    mx.eval(model.parameters())
    logger.info(f"Quantization complete in {time.time() - t0:.1f}s")

    cache_path.mkdir(parents=True, exist_ok=True)
    _mlx_save(str(cache_path), model)
    tokenizer.save_pretrained(str(cache_path))

    config_file = cache_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        cfg.setdefault("quantization", {"group_size": 64, "bits": bits})
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2)
    logger.info(f"Cached at {cache_path}")

    return model, tokenizer


def _load_local_dir(model_dir: Path):
    """Load from a local directory, auto-detecting the format."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")

    onebit_config = model_dir / "onebit_config.json"
    if onebit_config.exists():
        with open(onebit_config) as f:
            ob = json.load(f)
        if ob.get("quantization") == "ternary_1.58bit" or ob.get("format") == "ternary":
            return _load_ternary_local(model_dir)

    # Default: mlx-lm handles both quantized and FP16 checkpoints.
    import mlx_lm
    return mlx_lm.load(str(model_dir))


def _load_ternary_local(model_dir: Path):
    """Load an onebit-converted ternary model into the custom transformer."""
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
    """Download a model snapshot from the HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    return Path(local_dir)


def _load_safetensors(model_dir: Path) -> dict[str, mx.array]:
    """Load and merge all safetensors files in a directory."""
    import glob

    weights = {}
    for path in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        weights.update(mx.load(path))
    return weights

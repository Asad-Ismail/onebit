"""Model loading, quantization, and caching.

Registry names and HF repos are downloaded as FP16 and quantized to 4-bit with
MLX-LM's engine on first run, then cached under ``~/.cache/onebit``. Local
directories are loaded directly with mlx-lm (quantized or FP16).
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
        import mlx_lm
        return mlx_lm.load(model_name_or_path)
    else:
        hf_repo = model_name_or_path
        cache_name = hf_repo.replace("/", "--")

    cached_path = CACHE_DIR / cache_name
    if cached_path.exists() and (cached_path / "config.json").exists():
        import mlx_lm
        logger.info(f"Loading cached model from {cached_path}")
        return mlx_lm.load(str(cached_path))

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

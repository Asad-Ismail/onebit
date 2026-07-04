"""Model registry — maps short names to HuggingFace repos and configs."""

MODELS: dict[str, dict] = {
    # Standard HF models, quantized to 4-bit with mlx-lm on first run.
    "phi-4-14b": {
        "hf_repo": "microsoft/phi-4",
        "model_type": "phi3",
        "native_ternary": False,
        "params": "14B",
        "ram_gb": 8.0,
        "description": "Microsoft Phi-4 14B, 4-bit mlx-lm.",
    },
    "qwen2.5-7b": {
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "qwen2",
        "native_ternary": False,
        "params": "7B",
        "ram_gb": 4.2,
        "description": "Qwen 2.5 7B Instruct, 4-bit mlx-lm.",
    },
    "qwen2.5-3b": {
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
        "model_type": "qwen2",
        "native_ternary": False,
        "params": "3B",
        "ram_gb": 1.8,
        "description": "Qwen 2.5 3B Instruct, 4-bit mlx-lm.",
    },
    "qwen2.5-coder-32b": {
        "hf_repo": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "model_type": "qwen2",
        "native_ternary": False,
        "params": "32B",
        "ram_gb": 18.0,
        "description": "Qwen 2.5 Coder 32B, 4-bit mlx-lm.",
    },
    "llama3.1-8b": {
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "model_type": "llama",
        "native_ternary": False,
        "params": "8B",
        "ram_gb": 4.8,
        "description": "Meta Llama 3.1 8B Instruct, 4-bit mlx-lm.",
    },
}


def get_model_info(name: str) -> dict:
    """Get model info by name, or raise if not found."""
    if name in MODELS:
        return MODELS[name]
    raise ValueError(
        f"Unknown model '{name}'. Available: {', '.join(MODELS.keys())}\n"
        f"Or provide a HuggingFace repo ID or local path directly."
    )


def list_models() -> list[dict]:
    """Return all registered models with their info."""
    return [{"name": k, **v} for k, v in MODELS.items()]

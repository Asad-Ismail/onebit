"""Model registry — maps short names to HuggingFace repos and configs."""

MODELS: dict[str, dict] = {
    # --- Natively trained ternary (best quality) ---
    "bitnet-2b": {
        "hf_repo": "microsoft/bitnet-b1.58-2B-4T",
        "model_type": "bitnet",
        "native_ternary": True,
        "params": "2B",
        "ram_gb": 0.5,
        "description": "Microsoft BitNet b1.58 2B-4T. Native ternary, highest quality.",
    },
    # --- Post-training quantized (we convert on first run) ---
    "phi-4-14b": {
        "hf_repo": "microsoft/phi-4",
        "model_type": "phi3",
        "native_ternary": False,
        "params": "14B",
        "ram_gb": 3.3,
        "description": "Microsoft Phi-4 14B quantized to 1.58-bit ternary.",
    },
    "qwen2.5-7b": {
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "qwen2",
        "native_ternary": False,
        "params": "7B",
        "ram_gb": 1.7,
        "description": "Qwen 2.5 7B Instruct quantized to 1.58-bit ternary.",
    },
    "qwen2.5-3b": {
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
        "model_type": "qwen2",
        "native_ternary": False,
        "params": "3B",
        "ram_gb": 0.8,
        "description": "Qwen 2.5 3B Instruct quantized to 1.58-bit ternary.",
    },
    "qwen2.5-coder-32b": {
        "hf_repo": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "model_type": "qwen2",
        "native_ternary": False,
        "params": "32B",
        "ram_gb": 7.5,
        "description": "Qwen 2.5 Coder 32B — runs on 16GB Macs at 1.58-bit.",
    },
    "llama3.1-8b": {
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "model_type": "llama",
        "native_ternary": False,
        "params": "8B",
        "ram_gb": 1.9,
        "description": "Meta Llama 3.1 8B Instruct quantized to 1.58-bit ternary.",
    },
    # --- Pre-quantized ternary by third parties ---
    "falcon3-7b": {
        "hf_repo": "tiiuae/Falcon3-7B-Instruct-1.58bit",
        "model_type": "falcon",
        "native_ternary": False,
        "params": "7B",
        "ram_gb": 1.7,
        "description": "Falcon3 7B with QAT ternary by TII.",
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

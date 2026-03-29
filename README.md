# onebit

Run 32B models in 7.5 GB on your Mac. The first Metal-accelerated 1-bit LLM engine.

<!-- TODO: Add demo GIF here before launch -->

## Why?

| Model             | FP16   | 4-bit (ollama) | 1.58-bit (onebit) |
|-------------------|--------|----------------|--------------------|
| Qwen2.5 7B       | 14 GB  | 4.5 GB         | **1.7 GB**         |
| Phi-4 14B         | 28 GB  | 9.1 GB         | **3.3 GB**         |
| Qwen2.5-Coder 32B| 64 GB  | 20 GB          | **7.5 GB**         |

1-bit LLMs quantize every weight to {-1, 0, +1}. No multiplication needed in the matmul, just add and subtract. `onebit` runs them on your Mac's GPU via custom Metal compute shaders.

## Quickstart

```bash
pip install onebit
onebit run qwen2.5-3b
```

On first run, the model is downloaded from HuggingFace and quantized to 1.58-bit ternary. Subsequent runs load instantly from cache.

## Usage

```bash
# Interactive chat
onebit run qwen2.5-3b

# Single prompt
onebit run phi-4-14b -p "Write a Python function to find prime numbers"

# Benchmark
onebit bench qwen2.5-3b --runs 3

# Convert any HuggingFace model to ternary
onebit convert Qwen/Qwen2.5-3B-Instruct -o ./my-ternary-model
onebit run ./my-ternary-model

# List available models
onebit list
```

## Models

| Name               | Params | RAM    | Type            | Description                           |
|--------------------|--------|--------|-----------------|---------------------------------------|
| bitnet-2b          | 2B     | 0.5 GB | Native ternary  | Microsoft BitNet b1.58 2B-4T          |
| qwen2.5-3b        | 3B     | 0.8 GB | Converted       | Qwen 2.5 3B Instruct                  |
| qwen2.5-7b        | 7B     | 1.7 GB | Converted       | Qwen 2.5 7B Instruct                  |
| phi-4-14b          | 14B    | 3.3 GB | Converted       | Microsoft Phi-4 14B                   |
| qwen2.5-coder-32b | 32B    | 7.5 GB | Converted       | Qwen 2.5 Coder 32B                    |
| llama3.1-8b       | 8B     | 1.9 GB | Converted       | Meta Llama 3.1 8B Instruct            |

Or convert any model: `onebit convert <hf-repo> -o <output-dir>`

## Benchmarks

<!-- TODO: Fill with real benchmarks from M4 Pro before launch -->

Benchmarks run on Apple M4 Pro (48 GB):

| Model       | Tool       | Bits | Decode tok/s | Peak RAM |
|-------------|-----------|------|-------------|----------|
| Qwen2.5 3B | onebit    | 1.58 | TBD         | TBD      |
| Qwen2.5 3B | ollama Q4 | 4.0  | TBD         | TBD      |

## How It Works

1. **Ternary quantization**: Every weight is mapped to {-1, 0, +1} using absmean quantization from the BitNet b1.58 paper: `W_ternary = round(clip(W / mean(|W|), -1, 1))`. Weights are packed at 2 bits each (4 per byte).

2. **Custom Metal kernel**: For single-token decode, a Metal compute shader performs the matrix-vector multiply directly on packed ternary weights. Since weights are only {-1, 0, +1}, the operation is branchless conditional add/subtract — no floating-point multiplication on the weights.

3. **MLX backend**: Built on Apple's [MLX](https://github.com/ml-explore/mlx) framework for unified CPU/GPU memory on Apple Silicon. Prefill uses MLX's optimized matmul; decode uses our custom kernel.

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- No NVIDIA GPU needed

## Development

```bash
git clone git@github-personal:Asad-Ismail/onebit.git
cd onebit
pip install -e ".[dev]"
python -m pytest tests/
```

## License

MIT

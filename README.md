# onebit

Run any LLM on your Mac with aggressive quantization and Metal GPU acceleration. One command.

<!-- TODO: Add demo GIF here before launch -->

## Quickstart

```bash
pip install onebit
onebit run qwen2.5-3b
```

Downloads, quantizes to 4-bit, and runs on Metal GPU. Cached for instant subsequent loads.

## Usage

```bash
# Interactive chat
onebit run qwen2.5-7b

# Single prompt
onebit run qwen2.5-3b -p "Write a haiku about the ocean"

# Run ANY HuggingFace model (auto-quantizes on first run)
onebit run microsoft/Phi-3.5-mini-instruct -p "Explain gravity"

# Benchmark
onebit bench qwen2.5-3b --runs 3

# List pre-configured models
onebit list

# System info
onebit info
```

## Benchmarks

Apple M4 Pro (48 GB), 4-bit quantization, 128-token generation:

| Model | Params | Decode tok/s | Prefill tok/s | Peak RAM | TTFT |
|-------|--------|-------------|--------------|----------|------|
| Qwen2.5-3B | 3B | **119.8** | 350.9 | 1,750 MB | 131 ms |
| Phi-3.5-mini | 3.8B | **104.0** | 28.9 | 2,050 MB | -- |
| Qwen2.5-7B | 7B | **58.5** | 107.1 | 4,196 MB | 430 ms |

Any model on HuggingFace works. Just `onebit run <hf-repo>`.

## Pre-configured Models

```
$ onebit list
```

| Name | Params | Description |
|------|--------|-------------|
| qwen2.5-3b | 3B | Qwen 2.5 3B Instruct |
| qwen2.5-7b | 7B | Qwen 2.5 7B Instruct |
| phi-4-14b | 14B | Microsoft Phi-4 14B |
| qwen2.5-coder-32b | 32B | Qwen 2.5 Coder 32B |
| bitnet-2b | 2B | Microsoft BitNet b1.58 (native ternary) |

Or point to any HuggingFace repo directly: `onebit run Qwen/Qwen2.5-14B-Instruct`

## How It Works

1. **Auto-quantize**: Downloads any HuggingFace model, quantizes with MLX-LM's 4-bit engine (group_size=64), caches to `~/.cache/onebit/`

2. **Metal GPU inference**: All computation runs on Apple Silicon GPU via MLX. Unified memory means zero CPU-GPU transfer overhead.

3. **Custom ternary kernel**: For natively-trained 1-bit models (BitNet b1.58), a custom Metal compute shader performs branchless ternary matrix-vector multiply (conditional add/subtract, no multiplication).

4. **Streaming generation**: Live tok/s display, top-p sampling, chat template support for all major model families.

## Architecture

```
onebit/
  kernels/ternary.py   # Custom Metal shader for ternary GEMV
  layers.py            # BitLinear layer (packed 2-bit ternary weights)
  models/              # Transformer arch + config + model registry
  engine.py            # Download, quantize, cache, load
  generate.py          # Streaming generation with mlx-lm backend
  cli.py               # Click CLI with Rich output
  bench.py             # Benchmarking suite
```

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- No NVIDIA GPU needed

## Development

```bash
git clone https://github.com/Asad-Ismail/onebit.git
cd onebit
pip install -e ".[dev]"
python -m pytest tests/
```

## License

MIT

# onebit

Run LLMs locally on Apple Silicon in one command.

[![CI](https://github.com/Asad-Ismail/onebit/actions/workflows/ci.yml/badge.svg)](https://github.com/Asad-Ismail/onebit/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

onebit downloads a HuggingFace model, quantizes it, caches it, and streams
generation on the Metal GPU through [MLX](https://github.com/ml-explore/mlx).
No server to start, no config files, no CUDA — point it at a model name and go.

```console
$ onebit run qwen2.5-3b -p "In one sentence, what is MLX?"
Loaded qwen2.5-3b | RAM: 1656 MB | Device: Metal GPU

MLX is Apple's array framework for machine learning on Apple Silicon.
  24 tokens | prefill: 192.4 tok/s | decode: 116.6 tok/s | total: 1.3s
```

## Install

```bash
git clone https://github.com/Asad-Ismail/onebit.git
cd onebit
pip install -e .
```

Requires macOS on Apple Silicon (M1 or newer) and Python 3.10+.

## Usage

```bash
onebit run qwen2.5-7b                       # interactive chat
onebit run qwen2.5-3b -p "Explain gravity"  # single prompt
onebit run Qwen/Qwen2.5-14B-Instruct        # any HF repo, quantized on first run
onebit run ./my-model                       # a local directory

onebit bench qwen2.5-3b --runs 3            # tok/s, TTFT, peak RAM
onebit convert Qwen/Qwen2.5-3B-Instruct -o ./out   # export a ternary model
onebit list                                 # pre-configured models
onebit info                                 # system / MLX / Metal info
```

The first run of a model downloads and quantizes it into `~/.cache/onebit/`;
later runs load straight from cache.

## Benchmarks

Measured on an Apple M4 Pro (48 GB), 4-bit quantization, 128-token generation:

| Model      | Decode | Prefill | TTFT   | Peak RAM |
|------------|--------|---------|--------|----------|
| qwen2.5-3b | 116.6 tok/s | 192.4 tok/s | 239 ms | 1.75 GB |

Numbers depend on your chip, model, and prompt. Reproduce on your machine:

```bash
onebit bench qwen2.5-3b --runs 3
```

## Quantization

**4-bit (default).** Standard HuggingFace checkpoints are quantized with
MLX-LM's engine (`group_size=64`). This is the proven path and runs for every
model in the registry and any HF repo you pass.

**Ternary / 1.58-bit (experimental).** `onebit convert <repo> -o <dir>`
post-training-quantizes an FP16 checkpoint to ternary weights in onebit's own
format, which run on a custom MLX transformer with a Metal GEMV kernel that
replaces weight multiplies with conditional add/subtract. This path is not yet
quality-validated against the 4-bit path — treat it as experimental. Native
HuggingFace BitNet checkpoints (e.g. `microsoft/bitnet-b1.58-2B-4T`) use a
different on-disk packing and are not supported yet.

## Project layout

| Path | Purpose |
|------|---------|
| `engine.py` | Resolve, download, quantize, cache, and load models |
| `generate.py` | Streaming generation (mlx-lm backend + custom ternary path) |
| `quant.py` | Ternary pack/unpack and absmean quantization |
| `layers.py` | `BitLinear` — 2-bit packed ternary linear layer |
| `kernels/ternary.py` | Metal shader for ternary matrix-vector multiply |
| `models/` | Generic transformer, config mapping, model registry |
| `bench.py` | Benchmark harness |
| `cli.py` | Command-line interface |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check onebit
```

## License

MIT

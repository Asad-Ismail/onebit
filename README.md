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

Standard HuggingFace checkpoints are downloaded as FP16 and quantized to 4-bit
with MLX-LM's engine (`group_size=64`) on first run, then cached. Pass any HF
repo or a local model directory (already-quantized or FP16) — both load through
mlx-lm.

## Project layout

| Path | Purpose |
|------|---------|
| `engine.py` | Resolve, download, quantize, cache, and load models |
| `generate.py` | Streaming generation over the mlx-lm backend |
| `models/registry.py` | Pre-configured model registry |
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

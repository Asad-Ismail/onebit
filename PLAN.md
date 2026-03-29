# onebit — Implementation Plan

> Run 32B models in 7.5GB on your MacBook. First Metal-accelerated 1-bit LLM engine for Mac.

**Target:** 3K GitHub stars in first week
**Hardware:** MacBook Pro M4 Pro, 48GB RAM
**Timeline:** 10 days
**Date:** March 29, 2026

---

## 1. What We're Building

A polished, one-command tool for running ternary (1.58-bit) LLMs on Mac with Metal GPU acceleration. Think "ollama for 1-bit models."

```bash
pip install onebit
onebit run qwen2.5-coder-32b    # 32B model, 7.5GB RAM, Metal GPU
onebit run phi-4-14b             # 14B model, 3.3GB RAM
onebit run bitnet-2b             # Native ternary, 0.4GB
onebit bench phi-4-14b           # Benchmark with results table
onebit convert HF_MODEL_ID       # Convert any model to 1.58-bit
```

### Why This Is Novel (Not Duplicating Anything)

| Existing Tool         | Gap                                                            |
|-----------------------|----------------------------------------------------------------|
| bitnet.cpp (Microsoft)| CPU-only on Mac. Zero Metal GPU support. Only supports BitNet-2B well |
| mlx-lm                | Has 2-bit quants but does NOT support BitNet architecture (BitLinear, per-tensor activation quant) |
| bitnet-mlx-engine     | Tiny project (~0 stars), one model, no CLI, no model hub      |
| exo-explore/mlx-bitnet| Stale since March 2024, uses generic ops, no custom kernels   |
| tzervas ternary models| Raw weights on HF with no tool to run them                    |
| QVAC Fabric           | Crypto company project, LoRA-focused, not inference-optimized |
| llama.cpp             | TQ1_0/TQ2_0 have ZERO Metal shaders — ternary falls back to CPU |

**onebit fills the gap:** unified experience + Metal GPU + recent models + polished UX.

---

## 2. The Viral Hook

### Primary Headline
> "Run Qwen2.5-Coder 32B in 7.5GB on your MacBook — no NVIDIA needed"

### Why This Goes Viral
- **Surprising:** 32B model normally needs 64GB (FP16) or 18GB (4-bit). At 1.58-bit: 7.5GB.
- **Relatable hardware:** Runs on MacBooks people already own
- **Quantified claim:** Specific numbers (32B, 7.5GB, XX tok/s)
- **Democratization angle:** "No NVIDIA, no cloud, just your Mac"
- **Trending topic:** 1-bit LLMs are the hot research area of 2025-2026

### Alternative Headlines (pick based on benchmarks)
- "Run Phi-4 14B in 3.3GB on any MacBook"
- "1-bit inference on Mac GPU — 2x faster than CPU-only bitnet.cpp"
- "The smallest way to run a 32B model: 7.5GB with Metal acceleration"

---

## 3. Technical Architecture

### Stack
- **MLX** (v0.31+) for inference engine — handles Metal GPU, KV cache, sampling, tokenization
- **Custom Metal kernel** for ternary-optimized matmul via `mx.fast.metal_kernel`
- **Click + Rich** for CLI with live stats (tok/s, memory, model info)
- **HuggingFace Hub** for model downloads and hosting pre-converted models

### Why MLX (Not llama.cpp Fork)
1. MLX already runs on Metal GPU — GPU acceleration is free
2. Python = fast iteration, ship in days not months
3. `mx.fast.metal_kernel` gives raw Metal shader access with zero overhead
4. mlx-lm handles tokenization, sampling, KV cache — focus on ternary-specific parts
5. Community finding: C++ rewrite of MLX ternary engine gave 0% speedup — GPU kernel is the bottleneck, not host language
6. MLX v0.31 has 2/3/4/5/6/8-bit affine quant + FP4/FP8 — ternary is a targeted extension

### Project Structure
```
onebit/
├── onebit/
│   ├── __init__.py              # Version, public API
│   ├── cli.py                   # Click CLI with Rich output
│   ├── engine.py                # Inference engine orchestration
│   ├── generate.py              # Text generation (streaming, sampling)
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── ternary_matmul.metal # Custom Metal shader for ternary matmul
│   ├── layers/
│   │   ├── __init__.py
│   │   └── bitlinear.py         # BitLinear layer (ternary weights + per-tensor int8 act quant)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py          # Model name → HF repo + config mapping
│   │   ├── loader.py            # Load ternary weights into MLX model
│   │   └── architectures/       # Model architecture definitions
│   │       ├── __init__.py
│   │       ├── llama.py         # Llama/Qwen/Falcon architecture
│   │       ├── phi.py           # Phi-4 architecture
│   │       └── bitnet.py        # Microsoft BitNet architecture
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── ternary.py           # Ternary quantization: FP16 → {-1, 0, 1} + scale
│   │   └── pack.py              # Pack/unpack ternary weights (2-bit storage)
│   ├── convert.py               # HF model → ternary conversion pipeline
│   └── bench.py                 # Benchmarking suite (tok/s, memory, quality)
├── models/                      # Pre-converted model configs (YAML)
│   ├── bitnet-2b.yaml
│   ├── phi-4-14b.yaml
│   ├── qwen2.5-7b.yaml
│   └── qwen2.5-coder-32b.yaml
├── benchmarks/                  # Published benchmark results
│   └── m4_pro_48gb.md
├── tests/
│   ├── test_ternary_kernel.py
│   ├── test_bitlinear.py
│   └── test_generate.py
├── pyproject.toml
├── LICENSE                      # MIT
└── README.md
```

---

## 4. Core Technical Components

### 4.1 Custom Metal Kernel: Ternary MatVec

The key insight: ternary weights are {-1, 0, 1}. Matmul becomes conditional
add/subtract — no multiplication needed. This is simpler than generic 2-bit
dequant kernels.

```metal
#include <metal_stdlib>
using namespace metal;

// Ternary GEMV: output[row] = sum_k( weight[row,k] * input[k] )
// where weight[row,k] in {-1, 0, +1}
//
// Packing: 4 ternary weights per byte
//   00 = 0 (skip)
//   01 = +1 (add)
//   10 = -1 (subtract)
//
// Per-tensor scale applied after accumulation (one multiply at the end)

kernel void ternary_matvec(
    device const uint8_t* weights   [[buffer(0)]],   // packed ternary [M, K/4]
    device const half* input        [[buffer(1)]],   // activation [K]
    device half* output             [[buffer(2)]],   // result [M]
    device const half* scales       [[buffer(3)]],   // per-tensor scale [1] or per-row [M]
    constant uint& K                [[buffer(4)]],
    constant uint& M                [[buffer(5)]],
    uint3 tid      [[thread_position_in_threadgroup]],
    uint3 tgid     [[threadgroup_position_in_grid]],
    uint simd_lid  [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one output row
    const uint row = tgid.x;
    if (row >= M) return;

    const uint packed_K = K / 4;
    const uint weight_base = row * packed_K;

    float sum = 0.0f;

    // Each thread processes a strided chunk of the K dimension
    for (uint i = tid.x; i < packed_K; i += 256) {
        uint8_t pack = weights[weight_base + i];

        // Unpack 4 ternary weights and accumulate
        uint base_k = i * 4;
        for (uint j = 0; j < 4; j++) {
            uint8_t w = (pack >> (j * 2)) & 0x3;
            float x = float(input[base_k + j]);

            // Branchless: w=01 → +x, w=10 → -x, w=00 → 0
            float pos = float(w & 1);       // 1 if w=01
            float neg = float(w >> 1);      // 1 if w=10
            sum += (pos - neg) * x;
        }
    }

    // SIMD reduction across the threadgroup
    sum = simd_sum(sum);

    // First thread in each simdgroup writes partial sum to shared memory
    threadgroup float partial_sums[8];  // up to 8 simdgroups
    if (simd_lid == 0) {
        partial_sums[simd_gid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First thread does final reduction and applies scale
    if (tid.x == 0) {
        float total = 0.0f;
        uint num_simdgroups = min(256u / 32u, (packed_K + 255u) / 256u);
        for (uint s = 0; s < num_simdgroups; s++) {
            total += partial_sums[s];
        }
        output[row] = half(total * float(scales[0]));
    }
}
```

**Optimization opportunities (iterate after v1):**
- Vectorized loads: `uint4` to read 16 bytes (64 weights) at once
- Shared memory for input vector tiling (reduce global memory reads)
- Half2 SIMD: process 2 halfs simultaneously
- Separate prefill kernel (GEMM not GEMV) for prompt processing
- Handle TQ1_0 packing (1.69 bpw) in addition to 2-bit packing

### 4.2 BitLinear Layer

The core neural network layer for ternary models. Replaces nn.Linear.

```python
import mlx.core as mx
import mlx.nn as nn
from onebit.kernels import ternary_matvec_kernel

class BitLinear(nn.Module):
    """Linear layer with ternary weights {-1, 0, +1} and per-tensor activation quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Packed ternary weights: 4 weights per byte
        self.packed_weights = mx.zeros((out_features, in_features // 4), dtype=mx.uint8)
        # Per-tensor weight scale (from training: scale = mean(|W|))
        self.weight_scale = mx.array([1.0], dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        # Per-tensor activation quantization (matches BitNet b1.58 training)
        # Qb(x) = Clip(round(x * (127 / max(|x|))), -128, 127)
        act_scale = 127.0 / mx.maximum(mx.max(mx.abs(x)), 1e-5)
        x_quant = mx.clip(mx.round(x * act_scale), -128, 127).astype(mx.float16)

        # Ternary matmul via custom Metal kernel
        # For single token (decode): use GEMV kernel
        # For prompt (prefill): use tiled GEMM or fall back to MLX built-in
        if x_quant.shape[-2] == 1:
            out = ternary_matvec_kernel(self.packed_weights, x_quant.squeeze(-2), self.weight_scale)
            out = out.reshape(1, -1)
        else:
            # Prefill: unpack weights and use MLX matmul (Metal-accelerated)
            out = mx.matmul(x_quant, self._unpack_weights().T)
            out = out * self.weight_scale

        # Rescale: undo activation quantization, apply weight scale
        out = out / act_scale

        if hasattr(self, 'bias'):
            out = out + self.bias
        return out

    def _unpack_weights(self) -> mx.array:
        """Unpack 2-bit packed ternary weights to float16 {-1, 0, +1}."""
        packed = self.packed_weights  # [M, K/4] uint8
        # Unpack each byte into 4 ternary values
        w0 = (packed & 0x03).astype(mx.float16)
        w1 = ((packed >> 2) & 0x03).astype(mx.float16)
        w2 = ((packed >> 4) & 0x03).astype(mx.float16)
        w3 = ((packed >> 6) & 0x03).astype(mx.float16)
        # Decode: 01 → +1, 10 → -1, 00 → 0
        def decode(w):
            return (w & 1) - (w >> 1)  # Branchless: +1, -1, or 0
        unpacked = mx.concatenate([
            decode(w0)[..., None],
            decode(w1)[..., None],
            decode(w2)[..., None],
            decode(w3)[..., None],
        ], axis=-1).reshape(self.out_features, self.in_features)
        return unpacked
```

### 4.3 Weight Conversion Pipeline

```python
# onebit/convert.py — Convert any HuggingFace model to ternary

def quantize_to_ternary(weight: mx.array) -> tuple[mx.array, mx.array]:
    """
    Quantize FP16/BF16 weight matrix to ternary {-1, 0, +1}.

    Method: Absmean quantization (from BitNet b1.58 paper)
      scale = mean(|W|)
      W_ternary = round(clip(W / scale, -1, 1))

    Returns: (packed_weights [M, K/4] uint8, scale [1] float16)
    """
    scale = mx.mean(mx.abs(weight))
    normalized = weight / mx.maximum(scale, 1e-5)
    ternary = mx.clip(mx.round(normalized), -1, 1).astype(mx.int8)

    # Pack: encode -1 → 10, 0 → 00, +1 → 01, then pack 4 per byte
    encoded = mx.where(ternary == 1, 1, mx.where(ternary == -1, 2, 0)).astype(mx.uint8)
    M, K = encoded.shape
    encoded = encoded.reshape(M, K // 4, 4)
    packed = (encoded[..., 0] |
              (encoded[..., 1] << 2) |
              (encoded[..., 2] << 4) |
              (encoded[..., 3] << 6))
    return packed, scale.astype(mx.float16)


def convert_model(hf_model_id: str, output_path: str, calibration_data=None):
    """
    Convert a HuggingFace model to onebit ternary format.

    Steps:
    1. Download model from HuggingFace
    2. Quantize all linear layers to ternary
    3. Optionally: calibration-aware quantization (GPTQ-style)
    4. Save in onebit format (safetensors + config)
    """
    # Implementation:
    # - Load with mlx-lm or transformers
    # - Walk all nn.Linear layers
    # - Replace with quantize_to_ternary
    # - Save packed weights + scales + config
    pass
```

### 4.4 Model Registry

```python
# onebit/models/registry.py

MODELS = {
    # Natively trained ternary (best quality)
    "bitnet-2b": {
        "hf_repo": "microsoft/bitnet-b1.58-2B-4T",
        "architecture": "bitnet",
        "native_ternary": True,
        "params": "2B",
        "ram_estimate": "0.5GB",
        "description": "Microsoft's official 1-bit LLM. Native ternary, highest quality.",
    },

    # Pre-converted ternary models (hosted on our HF org)
    "phi-4-14b": {
        "hf_repo": "onebit-models/phi-4-14b-1.58bit",  # We host this
        "base_model": "microsoft/phi-4",
        "architecture": "phi",
        "native_ternary": False,
        "params": "14B",
        "ram_estimate": "3.3GB",
        "description": "Microsoft Phi-4 quantized to 1.58-bit ternary.",
    },

    "qwen2.5-7b": {
        "hf_repo": "onebit-models/qwen2.5-7b-instruct-1.58bit",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "architecture": "llama",  # Qwen2.5 uses llama-like arch
        "native_ternary": False,
        "params": "7B",
        "ram_estimate": "1.7GB",
        "description": "Qwen 2.5 7B Instruct quantized to 1.58-bit ternary.",
    },

    "qwen2.5-coder-32b": {
        "hf_repo": "onebit-models/qwen2.5-coder-32b-1.58bit",
        "base_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "architecture": "llama",
        "native_ternary": False,
        "params": "32B",
        "ram_estimate": "7.5GB",
        "description": "Qwen 2.5 Coder 32B — runs on 16GB Macs at 1.58-bit.",
    },

    "falcon3-7b": {
        "hf_repo": "tiiuae/Falcon3-7B-Instruct-1.58bit",
        "architecture": "llama",
        "native_ternary": False,  # QAT-converted by TII
        "params": "7B",
        "ram_estimate": "1.7GB",
        "description": "Falcon3 7B with QAT ternary quantization by TII.",
    },
}
```

### 4.5 CLI Interface

```python
# onebit/cli.py

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

@click.group()
def cli():
    """onebit — Run 1-bit LLMs on your Mac with Metal GPU."""
    pass

@cli.command()
@click.argument("model_name")
@click.option("--prompt", "-p", default=None, help="Single prompt (non-interactive)")
@click.option("--max-tokens", default=512, help="Max tokens to generate")
@click.option("--temp", default=0.7, help="Sampling temperature")
def run(model_name, prompt, max_tokens, temp):
    """Run a ternary model interactively."""
    console = Console()

    # Show model info
    console.print(f"[bold]Loading {model_name}...[/bold]")
    model, tokenizer = load_model(model_name)  # Auto-downloads from HF
    console.print(f"[green]Loaded.[/green] RAM: {get_memory_mb():.0f}MB | Device: Metal GPU")

    if prompt:
        # Single generation
        generate_and_stream(model, tokenizer, prompt, max_tokens, temp, console)
    else:
        # Interactive chat loop
        console.print("[dim]Type your message. Ctrl+C to exit.[/dim]\n")
        while True:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            generate_and_stream(model, tokenizer, user_input, max_tokens, temp, console)

@cli.command()
@click.argument("model_name")
def bench(model_name):
    """Benchmark a model: tok/s, memory, time-to-first-token."""
    # Run standardized benchmark and display Rich table
    pass

@cli.command()
@click.argument("hf_model_id")
@click.option("--output", "-o", required=True, help="Output directory")
@click.option("--calibration", default=None, help="Calibration dataset (optional)")
def convert(hf_model_id, output, calibration):
    """Convert any HuggingFace model to 1.58-bit ternary."""
    pass

@cli.command()
def list():
    """List available pre-converted models."""
    table = Table(title="Available Models")
    table.add_column("Name", style="bold")
    table.add_column("Params")
    table.add_column("RAM")
    table.add_column("Type")
    table.add_column("Description")
    for name, info in MODELS.items():
        table.add_row(
            name,
            info["params"],
            info["ram_estimate"],
            "Native" if info["native_ternary"] else "Converted",
            info["description"],
        )
    Console().print(table)
```

---

## 5. Model Strategy

### Models to Support at Launch (Priority Order)

| Priority | Model | Why | Action Needed |
|----------|-------|-----|---------------|
| P0 | Microsoft BitNet b1.58 2B-4T | Native ternary, guaranteed quality, MIT | Load from HF directly |
| P0 | Phi-4 14B 1.58-bit | Microsoft's latest, 14B in 3.3GB | Validate tzervas conversion OR convert ourselves |
| P0 | Qwen2.5-Coder-32B 1.58-bit | THE headline model — "32B on a MacBook" | Validate tzervas conversion OR convert ourselves |
| P1 | Qwen2.5-7B-Instruct 1.58-bit | Popular, good quality, smaller | Convert ourselves |
| P1 | Falcon3-7B-Instruct 1.58-bit | Already exists on HF from TII | Load from HF directly |
| P2 | User's own models via `onebit convert` | Key feature for power users | Build conversion pipeline |

### Quality Validation Plan

Before launch, we MUST validate output quality. Bad quality = no stars.

```bash
# For each model, run quick quality checks:
# 1. Perplexity on wikitext-2 (compare ternary vs 4-bit vs FP16)
# 2. MMLU 5-shot accuracy (the standard LLM benchmark)
# 3. Manual vibe check: 10 prompts, eyeball the responses
# 4. HumanEval pass@1 for coding models (Qwen-Coder)

# If ternary quality is within 5% of 4-bit → lead with quality story
# If ternary quality is 5-15% worse → lead with memory story, be honest about quality
# If ternary quality is >15% worse → don't include that model, focus on native ternary
```

### Conversion Quality Tiers

Label models honestly in the README:

| Tier | Label | Meaning |
|------|-------|---------|
| Gold | "Native Ternary" | Trained from scratch as ternary. Lossless. (BitNet 2B-4T) |
| Silver | "QAT Ternary" | Quantization-aware fine-tuned. Near-lossless. (Falcon3) |
| Bronze | "PTQ Ternary" | Post-training quantized. Some quality loss. (Our conversions) |

---

## 6. Benchmark Plan

### What to Measure

| Metric | How | Why |
|--------|-----|-----|
| Decode tok/s | Generate 256 tokens, measure wall time | Speed comparison |
| Prefill tok/s | Process 512-token prompt, measure time | Prompt processing speed |
| Time-to-first-token | Time from prompt to first output token | Responsiveness |
| Peak RAM | `mx.metal.get_active_memory()` during generation | Memory story |
| Model load time | Time from `onebit run` to ready | UX quality |

### Comparisons (Run All On Your M4 Pro 48GB)

| Tool | Config | Models to Test |
|------|--------|---------------|
| **onebit** (ours) | Metal GPU, 1.58-bit ternary | All models |
| **bitnet.cpp** | CPU-only, 1.58-bit ternary | BitNet 2B-4T |
| **mlx-lm** | Metal GPU, 4-bit quant | Phi-4, Qwen2.5-7B, Qwen2.5-32B |
| **ollama** | Metal GPU, Q4_K_M | Phi-4, Qwen2.5-7B |

### Expected Results (Rough Estimates for M4 Pro)

| Model | onebit (1.58-bit) | mlx-lm (4-bit) | ollama (Q4) | bitnet.cpp (CPU) |
|-------|-------------------|-----------------|-------------|------------------|
| 2B (BitNet) | ~80 tok/s | N/A | N/A | ~40 tok/s |
| 7B | ~35 tok/s | ~40 tok/s | ~35 tok/s | ~18 tok/s |
| 14B (Phi-4) | ~20 tok/s | ~22 tok/s | ~20 tok/s | ~10 tok/s |
| 32B | ~10 tok/s | OOM? / ~8 tok/s | OOM? | ~5 tok/s |
| **RAM (14B)** | **3.3GB** | **9.1GB** | **9.1GB** | **3.3GB** |
| **RAM (32B)** | **7.5GB** | **20GB** | **20GB** | **7.5GB** |

The story: Similar speed to 4-bit, but **2.5-3x less RAM**. And for 32B, we might be the ONLY tool that fits it comfortably on 16GB Macs.

---

## 7. Implementation Timeline

### Phase 1: Core Engine (Days 1-4)

#### Day 1: Foundation
- [ ] Set up project structure, pyproject.toml, dependencies
- [ ] Implement ternary weight packing/unpacking (`quant/ternary.py`, `quant/pack.py`)
- [ ] Implement BitLinear layer with basic MLX matmul (no custom kernel yet)
- [ ] Load BitNet b1.58 2B-4T weights and get first token generated
- **Milestone:** BitNet-2B generates text on Metal GPU

#### Day 2: Custom Metal Kernel
- [ ] Write `ternary_matmul.metal` (GEMV for decode)
- [ ] Integrate via `mx.fast.metal_kernel`
- [ ] Benchmark: custom kernel vs MLX built-in 2-bit vs CPU
- [ ] Iterate on kernel optimizations (vectorized loads, SIMD reduction)
- **Milestone:** Custom kernel working, initial benchmark numbers

#### Day 3: Model Architectures
- [ ] Implement Phi-4 architecture with BitLinear layers
- [ ] Implement Llama/Qwen architecture with BitLinear layers
- [ ] Write model loader that reads HF safetensors and packs to ternary
- [ ] Test: load and run Phi-4 14B in ternary mode
- **Milestone:** Phi-4 14B generates text at 1.58-bit

#### Day 4: Conversion Pipeline
- [ ] Implement `onebit convert` — FP16/BF16 → ternary quantization
- [ ] Add optional calibration (GPTQ-style with calibration dataset)
- [ ] Convert Phi-4 14B and Qwen2.5-7B, upload to HuggingFace
- [ ] Validate tzervas/qwen2.5-coder-32b-bitnet model quality
- [ ] If tzervas quality is bad, convert Qwen2.5-Coder-32B ourselves
- **Milestone:** 3+ models available on HuggingFace

### Phase 2: Polish & Benchmarks (Days 5-7)

#### Day 5: CLI + UX
- [ ] Build CLI with Click + Rich (run, bench, convert, list commands)
- [ ] Streaming text generation with live tok/s counter
- [ ] Interactive chat mode with conversation history
- [ ] Auto-download models from HuggingFace on first run
- [ ] Proper error messages (not enough RAM, model not found, etc.)
- **Milestone:** `pip install -e . && onebit run phi-4-14b` works end-to-end

#### Day 6: Benchmarks
- [ ] Run full benchmark suite on M4 Pro 48GB
- [ ] Compare: onebit vs bitnet.cpp vs mlx-lm 4-bit vs ollama Q4
- [ ] Measure: tok/s, memory, TTFT, prefill speed
- [ ] Run quality evals: perplexity, MMLU (at minimum), HumanEval for Coder
- [ ] Write up results in benchmarks/m4_pro_48gb.md
- **Milestone:** All benchmark numbers in hand

#### Day 7: Quality & Edge Cases
- [ ] Test on all supported models end-to-end
- [ ] Fix any generation quality issues (sampling, KV cache, chat template)
- [ ] Test conversion pipeline on 2-3 additional models
- [ ] Handle edge cases: 8GB Mac warnings, long prompts, special tokens
- **Milestone:** All models generate high-quality output reliably

### Phase 3: Launch Prep (Days 8-9)

#### Day 8: Demo & README
- [ ] Record demo GIF with asciinema or screen capture:
  - Show `onebit run qwen2.5-coder-32b`
  - Streaming code generation, live tok/s + memory display
  - ~10 seconds, high quality
- [ ] Write README (see template in Section 8)
- [ ] Create benchmark comparison table with real numbers
- [ ] Upload pre-converted models to HuggingFace
- [ ] Make `pip install onebit` work from PyPI
- **Milestone:** README is compelling, demo GIF is stunning

#### Day 9: Pre-Launch
- [ ] Test `pip install onebit && onebit run bitnet-2b` on a clean venv
- [ ] If possible, test on a different Mac (friend's M1/M2/M3)
- [ ] Write a short technical thread (for X/Twitter) explaining how it works
- [ ] DM 3-5 AI/ML Twitter accounts: "I'm launching this tomorrow, would love your take"
- [ ] Prepare HN post title and description
- [ ] Prepare r/LocalLLaMA post with benchmark table
- **Milestone:** Everything ready to launch

### Phase 4: Launch (Day 10 — MUST be Tuesday or Wednesday)

See Section 9.

---

## 8. README Template (Fill with Real Numbers)

```markdown
# onebit

Run 32B models in 7.5GB on your Mac. The first Metal-accelerated 1-bit LLM engine.

[DEMO GIF HERE]

## Why?

| Model | FP16 | 4-bit (ollama) | **1.58-bit (onebit)** |
|-------|------|----------------|----------------------|
| Phi-4 14B | 28 GB | 9.1 GB | **3.3 GB** |
| Qwen2.5-Coder 32B | 64 GB | 20 GB | **7.5 GB** |

1-bit LLMs quantize every weight to {-1, 0, 1}. No multiplication needed —
just add and subtract. `onebit` runs them on your Mac's GPU via Metal.

## Quickstart

    pip install onebit
    onebit run phi-4-14b

## Benchmarks (M4 Pro, 48GB)

| Model | Tool | Bits | tok/s | RAM | MMLU |
|-------|------|------|-------|-----|------|
| Phi-4 14B | **onebit** | 1.58 | XX | 3.3 GB | XX.X |
| Phi-4 14B | ollama | 4 | XX | 9.1 GB | XX.X |
| Phi-4 14B | bitnet.cpp | 1.58 | XX | 3.3 GB | XX.X |
| Qwen2.5-Coder 32B | **onebit** | 1.58 | XX | 7.5 GB | XX.X |
| Qwen2.5-Coder 32B | ollama | 4 | XX | 20 GB | XX.X |

## Models

    onebit list

| Name | Params | RAM | Type |
|------|--------|-----|------|
| bitnet-2b | 2B | 0.5 GB | Native ternary |
| phi-4-14b | 14B | 3.3 GB | Converted |
| qwen2.5-7b | 7B | 1.7 GB | Converted |
| qwen2.5-coder-32b | 32B | 7.5 GB | Converted |
| falcon3-7b | 7B | 1.7 GB | QAT |

## Convert Any Model

    onebit convert Qwen/Qwen2.5-7B-Instruct -o ./my-model
    onebit run ./my-model

## How It Works

Custom Metal compute shaders exploit ternary weight structure:
weights are {-1, 0, +1}, so matmul becomes conditional add/subtract.
Built on [MLX](https://github.com/ml-explore/mlx).

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- That's it. No NVIDIA GPU needed.
```

---

## 9. Launch Playbook

### HN Title Options (pick one based on benchmarks)

Option A (memory angle):
> Show HN: Onebit – Run Qwen2.5 32B in 7.5GB on your MacBook with Metal

Option B (speed angle):
> Show HN: Onebit – 1-bit LLM inference on Mac GPU, 2x faster than CPU

Option C (accessibility angle):
> Show HN: Onebit – Run 14B LLMs in 3GB RAM on any Mac, no NVIDIA needed

### Launch Schedule (Tuesday or Wednesday)

| Time (ET) | Action |
|-----------|--------|
| 8:00 AM | Submit "Show HN" post on news.ycombinator.com |
| 8:00 AM | Post on r/LocalLLaMA with benchmarks and demo GIF |
| 8:05 AM | Tweet/X post with demo GIF + link to HN + link to repo |
| 8:05 AM | Signal your prepared people to engage |
| 8:00-12:00 PM | **Answer every single comment on HN and Reddit** |
| 12:00 PM | If trending: cross-post to r/MachineLearning |
| 2:00 PM | Post on MLX Discord, llama.cpp Discord, LocalAI Discord |
| Evening | LinkedIn post (professional angle) |

### What to Say in Comments

On HN, be technical and humble:
- Explain the ternary matmul insight (no multiplication, just add/subtract)
- Share honest quality comparison numbers
- Acknowledge limitations ("post-training ternary quantization does lose some quality vs 4-bit")
- Talk about what's next ("working on prefill optimization, more models")

On Reddit, be practical:
- Share exact hardware and settings
- Help people troubleshoot installation
- Post comparison screenshots

### Post-Launch (Days 11-14)

- [ ] Ship a visible improvement within 48 hours (new model, speed bump, or new feature)
- [ ] Write a follow-up r/LocalLLaMA post: "onebit update: added X model, Y% faster"
- [ ] If you hit GitHub Trending, add a note to README
- [ ] Respond to all GitHub issues within 24 hours
- [ ] Record a 2-minute YouTube walkthrough

---

## 10. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Ternary quality is noticeably worse than 4-bit | High | High | Lead with memory story. Be transparent. Include native ternary model (BitNet-2B) as quality reference. Label models with quality tiers (Gold/Silver/Bronze) |
| Speed isn't faster than bitnet.cpp CPU | Medium | Medium | Focus on: (1) GPU leaves CPU free for other tasks, (2) prefill is much faster on GPU, (3) UX is better. May not need to beat on raw decode tok/s |
| MLX custom kernel doesn't help vs built-in 2-bit | High | Low | Use MLX built-in 2-bit as fallback. The kernel is a nice-to-have; the VALUE is in the UX and model support |
| Someone launches something similar this week | Low | High | Ship fast. First mover with polish beats late mover. 10-day timeline is aggressive but doable |
| pip install fails for users | Medium | Very High | Test on clean venv. Pin MLX version. Have clear error messages. Test on macOS 14/15/16 if possible |
| 32B model quality is garbage | Medium | High | Have Phi-4 14B as backup headline. 14B in 3.3GB is still very compelling |

---

## 11. Future Roadmap (After Launch)

These are not for v1, but signal ambition in the README:

- [ ] **Speculative decoding:** Use BitNet-2B as draft model for larger ternary models
- [ ] **Sparse-ternary:** Implement Sparse-BitNet (March 2026 paper) for even more compression
- [ ] **LoRA fine-tuning:** Train LoRA adapters on ternary models via MLX
- [ ] **Batch inference:** Support multiple concurrent requests
- [ ] **ollama-compatible API:** `onebit serve` with OpenAI-compatible endpoint
- [ ] **iOS/iPadOS:** MLX runs on mobile — ternary models are small enough for phones
- [ ] **More models:** Track every new ternary model release, add to registry
- [ ] **llama.cpp Metal PR:** Upstream the Metal shaders for TQ1_0/TQ2_0

---

## 12. Dependencies

```toml
# pyproject.toml
[project]
name = "onebit"
version = "0.1.0"
description = "Run 1-bit LLMs on your Mac with Metal GPU acceleration"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "mlx>=0.31.0",
    "mlx-lm>=0.31.0",
    "click>=8.0",
    "rich>=13.0",
    "huggingface-hub>=0.20",
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    "transformers>=4.40",
    "pyyaml>=6.0",
]

[project.scripts]
onebit = "onebit.cli:cli"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"
```

---

## Summary

**What:** A polished CLI tool that runs ternary (1.58-bit) LLMs on Mac with Metal GPU.
**Why viral:** "32B model in 7.5GB on your MacBook" — first time possible, great demo.
**How:** MLX + custom Metal kernel + pre-converted models + beautiful CLI.
**Timeline:** 10 days (4 days core, 3 days polish, 2 days prep, 1 day launch).
**Target:** 3K stars in first week via HN front page + r/LocalLLaMA + X/Twitter.

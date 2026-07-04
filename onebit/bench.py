"""Benchmarking utilities for onebit models."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from onebit.generate import generate_stream


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    prefill_tps: float
    decode_tps: float
    time_to_first_token_ms: float
    peak_memory_mb: float
    prompt_tokens: int
    generated_tokens: int
    total_time_s: float

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "prefill_tok/s": f"{self.prefill_tps:.1f}",
            "decode_tok/s": f"{self.decode_tps:.1f}",
            "TTFT_ms": f"{self.time_to_first_token_ms:.0f}",
            "peak_RAM_MB": f"{self.peak_memory_mb:.0f}",
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_time_s": f"{self.total_time_s:.2f}",
        }


# Standard benchmark prompts
BENCH_PROMPTS = [
    "Write a Python function that checks if a number is prime and explain how it works.",
    "Explain the theory of general relativity in simple terms.",
    "Write a short story about a robot who discovers music for the first time.",
]


def get_memory_mb() -> float:
    """Get current Metal GPU memory usage in MB."""
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def get_peak_memory_mb() -> float:
    """Get peak Metal GPU memory usage in MB."""
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except Exception:
        try:
            return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def reset_peak_memory() -> None:
    """Reset peak memory tracking."""
    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def benchmark_model(
    model,
    tokenizer,
    model_name: str = "unknown",
    prompt: str | None = None,
    max_tokens: int = 128,
    num_runs: int = 1,
) -> BenchmarkResult:
    """Run a benchmark on a model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        model_name: Name for display
        prompt: Custom prompt (defaults to standard bench prompt)
        max_tokens: Max tokens to generate
        num_runs: Number of runs to average

    Returns:
        BenchmarkResult with timing and memory stats
    """
    if prompt is None:
        prompt = BENCH_PROMPTS[0]

    best_decode_tps = 0
    best_result = None

    for run_idx in range(num_runs):
        reset_peak_memory()

        stats = None
        for _, stats in generate_stream(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,  # Greedy for reproducibility
        ):
            pass

        if stats is None:
            continue

        peak_mem = get_peak_memory_mb()

        result = BenchmarkResult(
            model_name=model_name,
            prefill_tps=stats.prefill_tps,
            decode_tps=stats.decode_tps,
            time_to_first_token_ms=stats.prefill_time_s * 1000,
            peak_memory_mb=peak_mem,
            prompt_tokens=stats.prompt_tokens,
            generated_tokens=stats.generated_tokens,
            total_time_s=stats.total_time_s,
        )

        if result.decode_tps > best_decode_tps:
            best_decode_tps = result.decode_tps
            best_result = result

    return best_result


def format_benchmark_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a markdown table."""
    lines = [
        "| Model | Prefill tok/s | Decode tok/s | TTFT (ms) | Peak RAM (MB) | Generated |",
        "|-------|--------------|-------------|-----------|--------------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r.model_name} | {r.prefill_tps:.1f} | {r.decode_tps:.1f} | "
            f"{r.time_to_first_token_ms:.0f} | {r.peak_memory_mb:.0f} | "
            f"{r.generated_tokens} tokens |"
        )
    return "\n".join(lines)

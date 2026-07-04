"""Text generation with streaming support (mlx-lm backend)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class GenerationStats:
    """Statistics from a generation run."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.prefill_time_s if self.prefill_time_s > 0 else 0

    @property
    def decode_tps(self) -> float:
        return self.generated_tokens / self.decode_time_s if self.decode_time_s > 0 else 0

    @property
    def total_time_s(self) -> float:
        return self.prefill_time_s + self.decode_time_s


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Apply the chat template if available, else plain encode."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            if hasattr(token_ids, "input_ids"):
                token_ids = token_ids.input_ids
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            return list(token_ids)
        except Exception:
            pass
    return list(tokenizer.encode(prompt))


def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[tuple[str, GenerationStats], None, None]:
    """Stream tokens from the model, yielding (new_text, stats) tuples."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    token_ids = _encode_prompt(tokenizer, prompt)
    stats = GenerationStats(prompt_tokens=len(token_ids))
    sampler = make_sampler(temp=temperature, top_p=top_p)

    t0 = time.perf_counter()
    decode_start: Optional[float] = None

    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=token_ids, max_tokens=max_tokens, sampler=sampler
    ):
        if decode_start is None:
            stats.prefill_time_s = time.perf_counter() - t0
            decode_start = time.perf_counter()

        stats.generated_tokens += 1
        stats.decode_time_s = time.perf_counter() - decode_start

        if response.text:
            yield response.text, stats


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, GenerationStats]:
    """Generate a complete response (non-streaming)."""
    chunks = []
    stats = None
    for text, stats in generate_stream(
        model, tokenizer, prompt, max_tokens, temperature, top_p
    ):
        chunks.append(text)
    return "".join(chunks), stats or GenerationStats()

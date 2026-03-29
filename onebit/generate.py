"""Text generation with streaming support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, Optional

import mlx.core as mx

from onebit.models.transformer import TransformerModel, KVCache
import mlx.nn as nn


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


def sample_token(logits: mx.array, temperature: float = 0.7, top_p: float = 0.9) -> mx.array:
    """Sample a token from logits with temperature and top-p."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_p < 1.0:
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        cumsum = mx.cumsum(probs, axis=-1)
        mask = cumsum - probs > top_p
        sorted_logits = mx.where(mask, mx.array(-1e9), sorted_logits)
        probs = mx.softmax(sorted_logits, axis=-1)
        sampled = mx.random.categorical(mx.log(probs + 1e-10))
        token = mx.take_along_axis(sorted_indices, sampled.reshape(-1, 1), axis=-1)
        return token.squeeze(-1)

    return mx.random.categorical(logits)


def generate_stream(
    model: TransformerModel,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_tokens: Optional[list[int]] = None,
) -> Generator[tuple[str, GenerationStats], None, None]:
    """Stream tokens from the model, yielding (new_text, stats) tuples.

    Usage:
        for text, stats in generate_stream(model, tokenizer, "Hello"):
            print(text, end="", flush=True)
        print(f"\\n[{stats.decode_tps:.1f} tok/s]")
    """
    # Encode prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            token_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            # Ensure we have a plain list of ints
            if hasattr(token_ids, "input_ids"):
                token_ids = token_ids.input_ids
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
        except Exception:
            token_ids = tokenizer.encode(prompt)
    else:
        token_ids = tokenizer.encode(prompt)

    # Ensure token_ids is a plain list of ints
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)

    input_ids = mx.array([token_ids])
    stats = GenerationStats(prompt_tokens=len(token_ids))

    # Determine stop tokens
    if stop_tokens is None:
        stop_tokens = []
        if hasattr(tokenizer, "eos_token_id"):
            eos = tokenizer.eos_token_id
            if isinstance(eos, int):
                stop_tokens.append(eos)
            elif isinstance(eos, list):
                stop_tokens.extend(eos)

    is_custom = isinstance(model, TransformerModel)

    if not is_custom:
        # Use mlx-lm's battle-tested generation for mlx-lm models
        yield from _generate_stream_mlxlm(
            model, tokenizer, token_ids, stats, max_tokens,
            temperature, top_p, stop_tokens,
        )
        return

    # === Custom ternary model path ===
    cache = model.make_cache()

    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    stats.prefill_time_s = time.perf_counter() - t0

    next_token = sample_token(logits[:, -1, :], temperature, top_p)
    mx.eval(next_token)
    token_id = next_token.item()

    if token_id in stop_tokens:
        return

    generated_ids = [token_id]
    prev_text = ""
    current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    new_text = current_text[len(prev_text):]
    prev_text = current_text
    stats.generated_tokens = 1

    if new_text:
        yield new_text, stats

    decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        input_ids = next_token.reshape(1, 1)
        logits = model(input_ids, cache=cache)
        next_token = sample_token(logits[:, -1, :], temperature, top_p)
        mx.eval(next_token)

        token_id = next_token.item()
        if token_id in stop_tokens:
            break

        generated_ids.append(token_id)
        stats.generated_tokens += 1
        stats.decode_time_s = time.perf_counter() - decode_start

        current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_text = current_text[len(prev_text):]
        prev_text = current_text

        if new_text:
            yield new_text, stats

    stats.decode_time_s = time.perf_counter() - decode_start


def _generate_stream_mlxlm(
    model, tokenizer, token_ids, stats, max_tokens,
    temperature, top_p, stop_tokens,
):
    """Generate using mlx-lm's native generation (handles KV cache correctly)."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    # Build sampler matching our temperature/top_p settings
    sampler = make_sampler(temp=temperature, top_p=top_p)

    t0 = time.perf_counter()
    first_token = True

    for response in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=token_ids,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if first_token:
            stats.prefill_time_s = time.perf_counter() - t0
            decode_start = time.perf_counter()
            first_token = False

        token_id = response.token
        if token_id in stop_tokens:
            break

        stats.generated_tokens += 1
        stats.decode_time_s = time.perf_counter() - decode_start

        new_text = response.text
        if new_text:
            yield new_text, stats


def generate(
    model: TransformerModel,
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

"""Tests for transformer model."""

import mlx.core as mx

from onebit.models.config import ModelConfig
from onebit.models.transformer import TransformerModel


def test_small_model_forward():
    """A tiny model should produce logits with correct shape."""
    config = ModelConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
    )

    model = TransformerModel(config, ternary=False)
    mx.eval(model.parameters())

    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids)
    mx.eval(logits)

    assert logits.shape == (1, 4, 256), f"Expected (1, 4, 256), got {logits.shape}"


def test_small_model_with_cache():
    """Model should work with KV cache for generation."""
    config = ModelConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
    )

    model = TransformerModel(config, ternary=False)
    mx.eval(model.parameters())

    cache = model.make_cache()

    # Prefill
    input_ids = mx.array([[1, 2, 3]])
    logits = model(input_ids, cache=cache)
    mx.eval(logits)

    assert logits.shape == (1, 3, 256)
    assert cache[0].offset == 3

    # Decode
    next_input = mx.array([[4]])
    logits = model(next_input, cache=cache)
    mx.eval(logits)

    assert logits.shape == (1, 1, 256)
    assert cache[0].offset == 4


def test_ternary_model():
    """Model with ternary=True should use BitLinear layers."""
    config = ModelConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
    )

    model = TransformerModel(config, ternary=True)
    mx.eval(model.parameters())

    input_ids = mx.array([[1, 2, 3]])
    logits = model(input_ids)
    mx.eval(logits)

    assert logits.shape == (1, 3, 256)


if __name__ == "__main__":
    test_small_model_forward()
    test_small_model_with_cache()
    test_ternary_model()
    print("All tests passed!")

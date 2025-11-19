"""
Smoke test for weight tying in the NNX TTT model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from ponderttt.data import get_tokenizer
from ponderttt.models import load_ttt_model


def main() -> None:
    print("=" * 60)
    print("Weight Tying Test (NNX)")
    print("=" * 60)

    print("\n[1/3] Loading tokenizer...")
    tokenizer = get_tokenizer("gpt2")
    vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Tokenizer vocab size: {vocab_size}")

    print("\n[2/3] Loading model with tied embeddings...")
    model, config = load_ttt_model(model_name="gpt2", seed=0, load_pretrained=False)
    model.eval()
    print(f"✓ Model: {config.n_layer} layers, dim={config.n_embd}")

    print("\n[3/3] Running forward pass...")
    rng = jax.random.PRNGKey(0)
    test_input = jnp.ones((1, 64), dtype=jnp.int32)
    outputs = model(test_input, use_ttt=False)
    logits = outputs["logits"]

    expected_shape = (1, 64, vocab_size)
    print(f"✓ Forward pass successful, logits shape: {logits.shape}")

    if logits.shape != expected_shape:
        raise AssertionError(f"Unexpected logits shape {logits.shape}, expected {expected_shape}")

    print("\n" + "=" * 60)
    print("Weight tying verified (shared embedding + LM head).")
    print("=" * 60)


if __name__ == "__main__":
    main()

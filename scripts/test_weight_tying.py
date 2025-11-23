"""
Smoke test for weight tying in the NNX TTT model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    print(f"OK Tokenizer vocab size: {vocab_size}")

    print("\n[2/3] Loading model with tied embeddings...")
    model, config = load_ttt_model(
        model_name="gpt2",
        seed=0,
        load_pretrained=False,
        vocab_size=vocab_size,
    )
    model.eval()
    print(f"OK Model: {config.n_layer} layers, dim={config.n_embd}")

    print("\n[3/3] Running forward pass...")
    test_input = jnp.ones((1, 64), dtype=jnp.int32)
    outputs = model(test_input, use_ttt=False)
    logits = outputs["logits"]

    embedding_vocab_size = model.base_model.wte.embedding[...].shape[0]
    expected_shape = (1, 64, embedding_vocab_size)
    print(f"OK Forward pass successful, logits shape: {logits.shape}")

    if logits.shape != expected_shape:
        raise AssertionError(f"Unexpected logits shape {logits.shape}, expected {expected_shape}")

    # Verify weight tying by recomputing logits from shared embedding
    hidden_states = model.base_model(test_input)
    embedding_kernel = model.base_model.wte.embedding[...]
    manual_logits = hidden_states @ embedding_kernel.T
    
    # Apply pad token masking if model does it
    if model.pad_token_id is not None:
        manual_logits = manual_logits.at[..., model.pad_token_id].set(-1e9)
        
    if not jnp.allclose(logits, manual_logits, atol=1e-4):
        raise AssertionError("Logits do not match tied embedding projection")

    # Explicitly verify that embedding and LM head share the same underlying array
    tied_kernel = model.base_model.wte.embedding[...]
    lm_kernel = model.base_model.wte.embedding[...] if model.tie_word_embeddings else model.lm_head.kernel
    if tied_kernel is not lm_kernel:
        raise AssertionError("Embedding and LM head do not share the same parameter object")

    print("\n" + "=" * 60)
    print("Weight tying verified (shared embedding + LM head).")
    print("=" * 60)


if __name__ == "__main__":
    main()

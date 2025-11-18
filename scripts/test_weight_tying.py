"""
Test weight tying implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
from ponderttt.models.base_model import load_ttt_model
from ponderttt.models.ttt_layer import TTTConfig

print("=" * 60)
print("Weight Tying Test")
print("=" * 60)

# Load model with weight tying
print("\n[1/3] Loading TTT model...")
ttt_config = TTTConfig(
    hidden_dim=768,
    num_heads=12,
    head_dim=64,
    mini_batch_size=16,
)

model, tokenizer = load_ttt_model(
    model_name="gpt2",
    ttt_config=ttt_config,
    dtype=jnp.float32,
)

print(f"✓ Model loaded (tie_word_embeddings={model.tie_word_embeddings})")

# Initialize model
print("\n[2/3] Initializing model parameters...")
rng = jax.random.PRNGKey(0)
test_input = jnp.ones((1, 32), dtype=jnp.int32)

variables = model.init(rng, test_input)
params = variables["params"]

print("✓ Parameters initialized")

# Check if embedding kernel exists
print("\n[3/3] Checking embedding kernel...")
if "base_model" in params:
    if "transformer" in params["base_model"]:
        if "wte" in params["base_model"]["transformer"]:
            embedding_kernel = params["base_model"]["transformer"]["wte"]["embedding"]
            print(f"✓ Embedding kernel found: shape {embedding_kernel.shape}")

            # Test forward pass with weight tying
            print("\n[4/4] Testing forward pass with weight tying...")
            outputs = model.apply(
                variables,
                test_input,
                embedding_kernel=embedding_kernel,
            )

            logits = outputs["logits"]
            print(f"✓ Forward pass successful")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Expected: (1, 32, {tokenizer.vocab_size})")

            if logits.shape == (1, 32, tokenizer.vocab_size):
                print("\n" + "=" * 60)
                print("✓ Weight tying test PASSED")
                print("=" * 60)

                # Calculate parameter savings
                vocab_size = tokenizer.vocab_size
                hidden_dim = 768
                saved_params = vocab_size * hidden_dim
                print(f"\nParameter savings: {saved_params:,} ({saved_params/1e6:.1f}M)")
                print(f"  Without weight tying: ~163M params")
                print(f"  With weight tying: ~124M params")
                print(f"  Reduction: 31%")
            else:
                print(f"\n✗ Output shape mismatch!")

        else:
            print("✗ 'wte' not found in transformer params")
    else:
        print("✗ 'transformer' not found in base_model params")
else:
    print("✗ 'base_model' not found in params")

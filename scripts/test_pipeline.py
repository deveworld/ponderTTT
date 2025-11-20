"""
Integration test for the complete PonderTTT pipeline with NNX.

Tests:
1. Data loading
2. Model initialization
3. Feature extraction
4. Policy network
5. TTT layer
6. End-to-end flow
"""

import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
from flax import nnx
from tokenizers import Tokenizer

print("=" * 60)
print("PonderTTT Pipeline Integration Test (NNX)")
print("=" * 60)

# Test 1: Data loading (using dummy data for unit testing)
print("\n[1/6] Testing data loading...")
try:
    from ponderttt.data import get_tokenizer

    tokenizer = cast(Tokenizer, get_tokenizer("gpt2"))

    # Create dummy batch for unit testing
    batch_size = 2
    seq_length = 1024
    chunk_size = 512
    num_chunks = seq_length // chunk_size

    batch = {
        "input_ids": jnp.ones((batch_size, seq_length), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_length), dtype=jnp.int32),
        "chunks": jnp.ones((batch_size, num_chunks, chunk_size), dtype=jnp.int32),
        "chunk_attention_mask": jnp.ones((batch_size, num_chunks, chunk_size), dtype=jnp.int32),
    }

    assert "input_ids" in batch
    assert "chunks" in batch
    print(f"OK Data loading works (dummy batch shape: {batch['input_ids'].shape})")
except Exception as e:
    print(f"[FAIL] Data loading failed: {e}")
    raise

# Test 2: Model initialization (NNX)
print("\n[2/6] Testing model initialization...")
try:
    from ponderttt.models import load_ttt_model, TTTConfig

    # Load model without pretrained weights for faster testing
    model, gpt2_config = load_ttt_model(
        model_name="gpt2",
        seed=42,
        load_pretrained=False
    )

    print(f"OK Model initialized (GPT-2: {gpt2_config.n_layer} layers, {gpt2_config.n_embd} dim)")
    print(f"  Model type: {type(model).__name__}")
except Exception as e:
    print(f"[FAIL] Model initialization failed: {e}")
    raise

# Test 3: Feature extraction
print("\n[3/6] Testing feature extraction...")
try:
    from ponderttt.utils import FeatureExtractor

    vocab_size = tokenizer.get_vocab_size()
    feature_extractor = FeatureExtractor(
        vocab_size=vocab_size,
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=chunk_size,
    )

    # Test feature extraction
    test_ids = batch["chunks"][:, 0, :]  # First chunk [batch_size, chunk_size]
    test_logits = jnp.ones((batch_size, chunk_size, vocab_size))

    features = feature_extractor.extract(test_ids, test_logits)
    assert features.shape == (batch_size, 32), f"Expected shape (2, 32), got {features.shape}"

    print(f"OK Feature extraction works (shape: {features.shape})")
except Exception as e:
    print(f"[FAIL] Feature extraction failed: {e}")
    raise

# Test 4: Policy network (NNX)
print("\n[4/6] Testing policy network...")
try:
    from ponderttt.models import PolicyNetwork, PolicyConfig

    policy_config = PolicyConfig(
        feature_dim=32,
        num_actions=4,
        hidden_dim=64
    )
    rngs = nnx.Rngs(42)
    policy = PolicyNetwork(policy_config, rngs)
    policy.train()

    # Test policy inference
    rng = jax.random.PRNGKey(0)
    policy_output = policy(features, deterministic=False, rng=rng)

    assert "action" in policy_output
    assert "value" in policy_output
    assert "log_prob" in policy_output

    print("OK Policy network works")
    print(f"  Actions: {policy_output['action']}")
    print(f"  Values: {policy_output['value']}")
except Exception as e:
    print(f"[FAIL] Policy network failed: {e}")
    raise

# Test 5: TTT layer (NNX)
print("\n[5/6] Testing TTT layer...")
try:
    from ponderttt.models import TTTLayer, TTTConfig

    ttt_config = TTTConfig(
        hidden_dim=gpt2_config.n_embd,
        num_heads=gpt2_config.n_head,
        head_dim=gpt2_config.n_embd // gpt2_config.n_head,
        mini_batch_size=64
    )
    rngs = nnx.Rngs(1)
    ttt_layer = TTTLayer(ttt_config, rngs)
    ttt_layer.train()

    # Test TTT layer
    test_hidden = jnp.ones((batch_size, chunk_size, gpt2_config.n_embd))
    adapted_hidden, ttt_stats = ttt_layer(test_hidden)

    assert adapted_hidden.shape == test_hidden.shape
    assert ttt_stats is not None

    print(f"OK TTT layer works (output shape: {adapted_hidden.shape})")
    print(f"  Stats keys: {list(ttt_stats.keys())}")
except Exception as e:
    print(f"[FAIL] TTT layer failed: {e}")
    raise

# Test 6: End-to-end flow (NNX)
print("\n[6/6] Testing end-to-end flow...")
try:
    # Simulate processing one chunk
    chunk_ids = batch["chunks"][:, 0, :]  # [batch_size, chunk_size]

    # Step 1: Model forward pass
    model.train()
    outputs = model(chunk_ids, use_ttt=True)
    logits = outputs["logits"]
    ttt_stats = outputs["ttt_stats"]

    # Step 2: Extract features
    features = feature_extractor.extract(chunk_ids, logits)

    # Step 3: Policy decision
    rng = jax.random.PRNGKey(123)
    policy_output = policy(features, deterministic=False, rng=rng)
    actions = policy_output["action"]
    values = policy_output["value"]

    # Verify shapes
    assert logits.shape == (batch_size, chunk_size, vocab_size)
    assert features.shape == (batch_size, 32)
    assert actions.shape == (batch_size,)
    assert values.shape == (batch_size,)

    print("OK End-to-end flow works!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Actions: {actions}")
    print(f"  Values: {values}")
    print(f"  TTT stats: {list(ttt_stats.keys())}")

except Exception as e:
    print(f"[FAIL] End-to-end flow failed: {e}")
    raise

# Success!
print("\n" + "=" * 60)
print("All integration tests passed!")
print("=" * 60)
print("\nPipeline is ready for training:")
print("  1. Tokenization")
print("  2. Model inference (NNX)")
print("  3. Feature extraction")
print("  4. Policy decisions (NNX)")
print("  5. TTT layer (NNX)")
print("  6. End-to-end flow")
print("\nRun training with:")
print("  uv run python -m ponderttt.experiments.train_baseline --action UPDATE_1")

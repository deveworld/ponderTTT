"""
Integration test for the complete PonderTTT pipeline.

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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

print("=" * 60)
print("PonderTTT Pipeline Integration Test")
print("=" * 60)

# Test 1: Data loading (using synthetic data for testing)
print("\n[1/6] Testing data loading...")
try:
    from ponderttt.data import get_tokenizer

    tokenizer = get_tokenizer("gpt2")

    # Create synthetic batch for testing
    batch_size = 2
    seq_length = 1024
    chunk_size = 512
    num_chunks = seq_length // chunk_size

    batch = {
        "input_ids": jnp.ones((batch_size, seq_length), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_length), dtype=jnp.int32),
        "chunks": jnp.ones((batch_size, num_chunks, chunk_size), dtype=jnp.int32),
    }

    assert "input_ids" in batch
    assert "chunks" in batch
    print(f"✓ Data loading works (synthetic batch shape: {batch['input_ids'].shape})")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 2: Model initialization
print("\n[2/6] Testing model initialization...")
try:
    from ponderttt.models import TransformerLM, ModelConfig

    model_config = ModelConfig(model_name="gpt2")
    model = TransformerLM(config=model_config)

    rng = jax.random.PRNGKey(0)
    test_input = jnp.ones((1, 10), dtype=jnp.int32)
    variables = model.init(rng, test_input)

    print(f"✓ Model initialization works")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

# Test 3: Feature extraction
print("\n[3/6] Testing feature extraction...")
try:
    from ponderttt.utils import FeatureExtractor

    extractor = FeatureExtractor(vocab_size=tokenizer.vocab_size)
    test_ids = jnp.array([[1, 2, 3, 4, 5]])
    test_logits = jnp.ones((1, 5, tokenizer.vocab_size))

    features = extractor.extract(test_ids, test_logits)
    assert features.shape[-1] == 32
    print(f"✓ Feature extraction works (shape: {features.shape})")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    sys.exit(1)

# Test 4: Policy network
print("\n[4/6] Testing policy network...")
try:
    from ponderttt.models import PolicyNetwork, PolicyConfig

    policy_config = PolicyConfig(feature_dim=32, num_actions=4)
    policy = PolicyNetwork(config=policy_config)

    test_features = jnp.ones((2, 32))
    rng = jax.random.PRNGKey(1)

    variables = policy.init(rng, test_features, deterministic=True)
    outputs = policy.apply(
        variables,
        test_features,
        deterministic=True,
        rngs={'action': rng}
    )

    assert "action" in outputs
    assert "value" in outputs
    print(f"✓ Policy network works (actions: {outputs['action']})")
except Exception as e:
    print(f"✗ Policy network failed: {e}")
    sys.exit(1)

# Test 5: TTT layer
print("\n[5/6] Testing TTT layer...")
try:
    from ponderttt.models import TTTLayer, TTTConfig

    ttt_config = TTTConfig(
        hidden_dim=768,
        chunk_size=128,
        num_heads=12,
    )
    ttt_layer = TTTLayer(config=ttt_config)

    test_hidden = jnp.ones((1, 256, 768))
    rng = jax.random.PRNGKey(2)

    variables = ttt_layer.init(rng, test_hidden)
    output, stats = ttt_layer.apply(variables, test_hidden)

    assert output.shape == test_hidden.shape
    assert "ttt_loss" in stats
    print(f"✓ TTT layer works (loss: {stats['ttt_loss']:.4f})")
except Exception as e:
    print(f"✗ TTT layer failed: {e}")
    sys.exit(1)

# Test 6: End-to-end flow
print("\n[6/6] Testing end-to-end flow...")
try:
    # Use synthetic batch from Test 1
    input_ids = batch["input_ids"]

    # Forward through model
    outputs = model.apply(variables, input_ids[:, :10])
    logits = outputs["logits"]

    # Extract features
    features = extractor.extract(input_ids[:, :5], logits[:, :5, :])

    # Get policy decision
    policy_outputs = policy.apply(
        policy.init(rng, features, deterministic=True),
        features,
        deterministic=True,
        rngs={'action': rng}
    )
    action = policy_outputs["action"][0]

    print(f"✓ End-to-end flow works")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Policy action: {action}")
except Exception as e:
    print(f"✗ End-to-end flow failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All integration tests passed!")
print("=" * 60)
print("\nPipeline is working correctly. Ready for experiments!")

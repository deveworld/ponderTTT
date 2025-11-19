"""
Quick test to verify JAX/Flax NNX implementation.
"""

import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
from flax import nnx
from tokenizers import Tokenizer

from ponderttt.data import get_tokenizer
from ponderttt.models import (
    PolicyNetwork,
    TTTLayer,
    PolicyConfig,
    TTTConfig,
    load_ttt_model,
)
from ponderttt.utils import FeatureExtractor

print("=" * 60)
print("PonderTTT JAX/Flax NNX Quick Test")
print("=" * 60)

# Check JAX
print(f"\nJAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# Track test results
tests_passed = 0
tests_failed = 0
total_tests = 5

# Test 1: Tokenizer
print("\n[1/5] Testing tokenizer...")
try:
    tokenizer = cast(Tokenizer, get_tokenizer("gpt2"))
    vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Tokenizer loaded (vocab size: {vocab_size})")
    tests_passed += 1
except Exception as e:
    print(f"✗ Tokenizer test failed: {e}")
    tests_failed += 1

# Test 2: TTT Transformer Model (NNX)
print("\n[2/5] Testing TTT transformer model...")
try:
    # Create model without loading pretrained weights (faster)
    model, config = load_ttt_model(
        model_name="gpt2",
        seed=42,
        load_pretrained=False
    )
    print("✓ TTT model created")

    # Forward pass
    test_input = jnp.ones((1, 64), dtype=jnp.int32)
    model.train()  # Set to training mode
    outputs = model(test_input, use_ttt=True)

    assert isinstance(outputs, dict), "Expected dict output from model"
    print(f"✓ Forward pass successful, logits shape: {outputs['logits'].shape}")
    print(f"  TTT stats: {list(outputs['ttt_stats'].keys())}")
    tests_passed += 1

except Exception as e:
    print(f"✗ TTT model test failed: {e}")
    tests_failed += 1

# Test 3: TTT Layer
print("\n[3/5] Testing TTT layer...")
try:
    ttt_config = TTTConfig(
        hidden_dim=768,
        num_heads=12,
        head_dim=64,
        mini_batch_size=16
    )
    rngs = nnx.Rngs(1)
    ttt_layer = TTTLayer(ttt_config, rngs)

    test_hidden = jnp.ones((1, 64, 768))
    ttt_layer.train()  # Set to training mode

    output, stats = ttt_layer(test_hidden)
    print(f"✓ TTT layer works, output shape: {output.shape}")
    if stats:
        print(f"  Stats keys: {list(stats.keys())}")
    tests_passed += 1

except Exception as e:
    print(f"✗ TTT layer test failed: {e}")
    tests_failed += 1

# Test 4: Policy Network (NNX)
print("\n[4/5] Testing policy network...")
try:
    policy_config = PolicyConfig(feature_dim=32, num_actions=4)
    rngs = nnx.Rngs(2)
    policy = PolicyNetwork(policy_config, rngs)

    test_features = jnp.ones((4, 32))
    policy.train()  # Set to training mode

    rng = jax.random.PRNGKey(42)
    policy_outputs = policy(test_features, deterministic=True, rng=rng)

    print("✓ Policy network works")
    print(f"  Actions: {policy_outputs['action']}")
    print(f"  Mean value: {jnp.mean(policy_outputs['value']):.4f}")
    tests_passed += 1

except Exception as e:
    print(f"✗ Policy network test failed: {e}")
    tests_failed += 1

# Test 5: Feature Extraction
print("\n[5/5] Testing feature extraction...")
try:
    tokenizer = cast(Tokenizer, get_tokenizer("gpt2"))
    vocab_size = tokenizer.get_vocab_size()
    extractor = FeatureExtractor(vocab_size=vocab_size)

    test_ids = jnp.array([[1, 2, 3, 4, 5]])
    test_logits = jnp.ones((1, 5, vocab_size))

    features = extractor.extract(test_ids, test_logits)
    print(f"✓ Feature extraction works, shape: {features.shape}")
    assert features.shape[-1] == 32, "Features should be 32-dimensional"
    tests_passed += 1

except Exception as e:
    print(f"✗ Feature extraction test failed: {e}")
    tests_failed += 1

# Summary
print("\n" + "=" * 60)
if tests_passed == total_tests:
    print(f"All {total_tests} tests passed!")
    print("=" * 60)
    print("\nJAX/Flax NNX implementation is ready.")
else:
    print(f"Tests: {tests_passed}/{total_tests} passed, {tests_failed} failed")
    print("=" * 60)
    print("\nSome tests failed. Please fix the issues above.")
    if tests_passed > 0:
        print(f"Working components: {tests_passed}/{total_tests}")

print("\nNext steps:")
print("  1. Test on TPU: gcloud compute tpus...")
print("  2. Run full training experiment")
print("  3. Train with: uv run python -m ponderttt.experiments.train_baseline --action UPDATE_1")

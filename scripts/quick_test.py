"""
Quick test to verify JAX/Flax implementation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
from ponderttt.data import get_tokenizer
from ponderttt.models import TransformerLM, PolicyNetwork, TTTLayer
from ponderttt.models import PolicyConfig, TTTConfig, ModelConfig
from ponderttt.utils import FeatureExtractor

print("=" * 60)
print("PonderTTT JAX/Flax Quick Test")
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
    tokenizer = get_tokenizer("gpt2")
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    tests_passed += 1
except Exception as e:
    print(f"✗ Tokenizer test failed: {e}")
    tests_failed += 1

# Test 2: Base Model
print("\n[2/5] Testing base model...")
try:
    model_config = ModelConfig(model_name="gpt2")
    model = TransformerLM(config=model_config)

    # Initialize
    rng = jax.random.PRNGKey(0)
    test_input = jnp.ones((1, 10), dtype=jnp.int32)

    variables = model.init(rng, test_input)
    print(f"✓ Base model initialized")

    # Forward pass
    outputs = model.apply(variables, test_input)
    print(f"✓ Forward pass successful, logits shape: {outputs['logits'].shape}")
    tests_passed += 1

except Exception as e:
    print(f"✗ Base model test failed: {e}")
    tests_failed += 1

# Test 3: TTT Layer
print("\n[3/5] Testing TTT layer...")
try:
    ttt_config = TTTConfig(hidden_dim=768, chunk_size=128)
    ttt_layer = TTTLayer(config=ttt_config)

    test_hidden = jnp.ones((2, 256, 768))
    rng = jax.random.PRNGKey(1)

    variables = ttt_layer.init(rng, test_hidden)
    output, stats = ttt_layer.apply(variables, test_hidden)
    print(f"✓ TTT layer works, output shape: {output.shape}")
    print(f"  TTT loss: {stats['ttt_loss']:.4f}")
    tests_passed += 1

except Exception as e:
    print(f"✗ TTT layer test failed: {e}")
    tests_failed += 1

# Test 4: Policy Network
print("\n[4/5] Testing policy network...")
try:
    policy_config = PolicyConfig(feature_dim=32, num_actions=4)
    policy = PolicyNetwork(config=policy_config)

    test_features = jnp.ones((4, 32))
    rng = jax.random.PRNGKey(2)

    variables = policy.init({'params': rng, 'dropout': rng}, test_features)
    policy_outputs = policy.apply(
        variables,
        test_features,
        rngs={'action': rng, 'dropout': rng}
    )

    print(f"✓ Policy network works")
    print(f"  Actions: {policy_outputs['action']}")
    print(f"  Mean value: {jnp.mean(policy_outputs['value']):.4f}")
    tests_passed += 1

except Exception as e:
    print(f"✗ Policy network test failed: {e}")
    tests_failed += 1

# Test 5: Feature Extraction
print("\n[5/5] Testing feature extraction...")
try:
    extractor = FeatureExtractor(vocab_size=tokenizer.vocab_size)

    test_ids = jnp.array([[1, 2, 3, 4, 5]])
    test_logits = jnp.ones((1, 5, tokenizer.vocab_size))

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
    print(f"✓ All {total_tests} tests passed!")
    print("=" * 60)
    print("\nJAX/Flax implementation is ready.")
else:
    print(f"Tests: {tests_passed}/{total_tests} passed, {tests_failed} failed")
    print("=" * 60)
    print(f"\n⚠️  Some tests failed. Please fix the issues above.")
    if tests_passed > 0:
        print(f"Working components: {tests_passed}/{total_tests}")

print("\nNext steps:")
print("  1. Test on TPU: gcloud compute tpus...")
print("  2. Run full training experiment")

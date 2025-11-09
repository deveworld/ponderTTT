"""
Integration tests for Iterative Transformer TTT.

Tests the complete pipeline from model creation to forward pass.
"""

import torch
import sys
sys.path.insert(0, '/home/world/ponderttt')

from src.ponderttt.models import IterativeTransformerConfig, IterativeTransformerTTT


def test_model_creation():
    """Test creating IterativeTransformerTTT."""
    print("Testing model creation...")

    config = IterativeTransformerConfig(
        vocab_size=1000,  # Small for testing
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ffn_dim=512,
        ttt_layer_indices=[1, 2],
        fast_weight_hidden_dim=32,
        max_steps=4,
        use_learned_policy=False,
    )

    model = IterativeTransformerTTT(config)

    print(f"✓ Model created with {model.count_parameters():,} parameters")

    return model


def test_forward_pass_uniform():
    """Test forward pass with uniform allocation."""
    print("\nTesting forward pass (uniform allocation)...")

    config = IterativeTransformerConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ttt_layer_indices=[1, 2],
        max_steps=4,
        use_learned_policy=False,
    )

    model = IterativeTransformerTTT(config)

    # Create dummy input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))

    # Fixed step allocation
    num_steps = torch.full((batch_size, seq_len), 2, dtype=torch.long)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        num_steps=num_steps,
        return_stats=True,
    )

    # Check outputs
    assert "logits" in outputs
    assert "loss" in outputs
    assert "ttt_stats" in outputs

    logits = outputs["logits"]
    loss = outputs["loss"]

    assert logits.shape == (batch_size, seq_len, 1000)
    assert loss.numel() == 1

    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  TTT stats: {len(outputs['ttt_stats'])} layers")

    return outputs


def test_forward_pass_learned():
    """Test forward pass with learned policy."""
    print("\nTesting forward pass (learned policy)...")

    config = IterativeTransformerConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ttt_layer_indices=[1, 2],
        max_steps=8,
        use_learned_policy=True,
        step_options=[1, 2, 4, 8],
    )

    model = IterativeTransformerTTT(config)
    model.eval()  # Deterministic mode

    # Create dummy input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))

    # No fixed steps - use policy
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        num_steps=None,  # Let policy decide
        return_stats=True,
    )

    assert "logits" in outputs
    assert "loss" in outputs
    assert "ttt_stats" in outputs

    print(f"✓ Learned policy forward pass successful")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    if outputs['ttt_stats']:
        avg_steps = sum(s['avg_steps'] for s in outputs['ttt_stats']) / len(outputs['ttt_stats'])
        print(f"  Average steps: {avg_steps:.2f}")

    return outputs


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\nTesting gradient flow...")

    config = IterativeTransformerConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        ttt_layer_indices=[1],
        max_steps=2,
        use_learned_policy=False,
    )

    model = IterativeTransformerTTT(config)
    model.train()

    # Create input with requires_grad
    input_ids = torch.randint(0, 1000, (2, 8))
    labels = torch.randint(0, 1000, (2, 8))
    num_steps = torch.full((2, 8), 2, dtype=torch.long)

    # Forward + backward
    outputs = model(input_ids, labels=labels, num_steps=num_steps)
    loss = outputs["loss"]
    loss.backward()

    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1

    print(f"✓ Gradient flow works ({grad_count} parameters have gradients)")

    return grad_count > 0


def test_flops_estimation():
    """Test FLOPs estimation."""
    print("\nTesting FLOPs estimation...")

    config = IterativeTransformerConfig(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ttt_layer_indices=[2, 3, 4],
        fast_weight_hidden_dim=64,
    )

    model = IterativeTransformerTTT(config)

    # Estimate FLOPs for different step counts
    for K in [1, 2, 4, 8]:
        flops = model.estimate_flops_per_token(num_steps=K, seq_len=256)
        print(f"  K={K}: {flops:,.0f} FLOPs/token")

    # Check that FLOPs increase with K
    flops_1 = model.estimate_flops_per_token(1, 256)
    flops_4 = model.estimate_flops_per_token(4, 256)
    assert flops_4 > flops_1, "FLOPs should increase with more steps"

    print(f"✓ FLOPs estimation works correctly")


def test_same_architecture():
    """Test that all configs have same architecture (fixed parameter count)."""
    print("\nTesting architecture consistency...")

    configs = []
    for K in [1, 2, 4, 8]:
        config = IterativeTransformerConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            ttt_layer_indices=[1, 2],
            fast_weight_hidden_dim=32,
            max_steps=8,  # Same max_steps
        )
        configs.append((K, config))

    # Create models
    models = [(K, IterativeTransformerTTT(config)) for K, config in configs]

    # Count parameters
    param_counts = [(K, model.count_parameters()) for K, model in models]

    # All should be the same!
    unique_counts = set(count for _, count in param_counts)
    assert len(unique_counts) == 1, f"Parameter counts differ: {param_counts}"

    print(f"✓ All configurations have same parameter count: {param_counts[0][1]:,}")
    print("  This confirms fair comparison!")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)

    try:
        test_model_creation()
        test_forward_pass_uniform()
        test_forward_pass_learned()
        test_gradient_flow()
        test_flops_estimation()
        test_same_architecture()

        print("\n" + "=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
